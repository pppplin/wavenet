import numpy as np
import tensorflow as tf

from .ops import causal_conv, mu_law_encode


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        return tf.Variable(initial_val, name=name)
    else:
        return create_variable(name, shape)


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


class WaveNetModel(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''
    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 quantization_channels=2**8,
                 use_biases=False,
                 scalar_input=False,
                 velocity_input=False,
                 midi_input=False,
                 initial_filter_width=32,
                 histograms=False,
                 load_chord=False,
                 chain_mel=False,
                 chain_vel=False,
                 global_condition_channels=None,
                 global_condition_cardinality=None,
                 local_condition_channels=None,
                 local_upsample_rate=None,
                 condition_restriction=None):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            scalar_input: Whether to use the quantized waveform directly as
                input to the network instead of one-hot encoding it.
                Default: False.
            velocity_input: Whether to include velocity in input. Default False
            initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input. This is only relevant
                if scalar_input=True.
            histograms: Whether to store histograms in the summary.
                Default: False.
            global_condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
            global_condition_cardinality: Number of mutually exclusive
                categories to be embedded in global condition embedding. If
                not None, then this implies that global_condition tensor
                specifies an integer selecting which of the N global condition
                categories, where N = global_condition_cardinality. If None,
                then the global_condition tensor is regarded as a vector which
                must have dimension global_condition_channels.
        '''
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.velocity_input = velocity_input
        self.midi_input = midi_input
        self.initial_filter_width = initial_filter_width
        self.histograms = histograms
        self.global_condition_channels = global_condition_channels
        self.global_condition_cardinality = global_condition_cardinality
        self.local_condition_channels = local_condition_channels
        self.local_upsample_rate = local_upsample_rate
        self.load_chord = load_chord
        self.chain_mel = chain_mel
        self.chain_vel = chain_vel
        self.condition_restriction = condition_restriction
        self.receptive_field = WaveNetModel.calculate_receptive_field(
            self.filter_width, self.dilations, self.scalar_input,
            self.initial_filter_width)
        self.variables = self._create_variables()

    @staticmethod
    def calculate_receptive_field(filter_width, dilations, scalar_input,
                                  initial_filter_width):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += filter_width - 1
        return receptive_field

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('wavenet'):
            if self.global_condition_cardinality is not None:
                # We only look up the embedding if we are conditioning on a
                # set of mutually-exclusive categories. We can also condition
                # on an already-embedded dense vector, in which case it's
                # given to us and we don't need to do the embedding lookup.
                # Still another alternative is no global condition at all, in
                # which case we also don't do a tf.nn.embedding_lookup.
                with tf.variable_scope('embeddings'):
                    layer = dict()
                    layer['gc_embedding'] = create_embedding_table(
                        'gc_embedding',
                        [self.global_condition_cardinality,
                         self.global_condition_channels])
                    if self.local_condition_channels is not None:
                        layer['lc_embedding'] = create_variable('lc_embedding', [1, 1, self.local_condition_channels, 1])
                    """TODO: remove manual assignment"""
                    if self.chain_mel or self.chain_vel:
                        layer['gc_vel'] = create_embedding_table('gc_vel', [4, self.global_condition_channels])
                        layer['gc_pitch'] = create_embedding_table('gc_pitch', [128, self.global_condition_channels])
                    var['embeddings'] = layer

            with tf.variable_scope('causal_layer'):
                layer = dict()
                if self.scalar_input:
                    initial_channels = 1
                    initial_filter_width = self.initial_filter_width
                else:
                    initial_channels = self.quantization_channels
                    initial_filter_width = self.filter_width
                layer['filter'] = create_variable(
                    'filter',
                    [initial_filter_width,
                     initial_channels,
                     self.residual_channels])
                layer['filter_velocity'] = create_variable(
                    'filter_velocity',
                    [initial_filter_width,
                    1,
                    initial_channels,
                    self.residual_channels])
                var['causal_layer'] = layer

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['filter_velocity'] = create_variable(
                                'filter_velocity',
                                [self.filter_width,
                                1,
                                self.residual_channels,
                                self.dilation_channels])
                        current['gate_velocity'] = create_variable(
                                'gate_velocity',
                                [self.filter_width,
                                1,
                                self.residual_channels,
                                self.dilation_channels])
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        current['dense_velocity'] = create_variable(
                                'dense_velocity',
                                [1, 1, self.dilation_channels, self.residual_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])
                        current['skip_velocity'] = create_variable(
                            'skip_velocity',
                            [1, 1, self.dilation_channels, self.skip_channels])
                        if self.global_condition_channels is not None:
                            current['gc_gateweights'] = create_variable(
                                'gc_gate',
                                [1, self.global_condition_channels,
                                 self.dilation_channels])
                            current['gc_filtweights'] = create_variable(
                                'gc_filter',
                                [1, self.global_condition_channels,
                                 self.dilation_channels])
                        if self.chain_mel and self.condition_restriction is not None \
                                and i<self.condition_restriction:
                            current['minor_gateweights'] = create_variable(
                                    'minor_gate',
                                    [1, self.global_condition_channels,
                                    self.dilation_channels])
                            current['minor_filtweights'] = create_variable(
                                    'minor_filter',
                                    [1, self.global_condition_channels,
                                    self.dilation_channels])
                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.skip_channels, self.skip_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.skip_channels, self.quantization_channels])
                current['postprocess1_velocity'] = create_variable(
                        'postprocess1_velocity',
                        [1, 1, self.skip_channels, self.skip_channels])
                current['postprocess2_velocity'] = create_variable(
                        'postprocess2_velocity',
                        [1, 1, self.skip_channels, self.quantization_channels])
                if self.use_biases:
                    current['postprocess1_bias'] = create_bias_variable(
                        'postprocess1_bias',
                        [self.skip_channels])
                    current['postprocess2_bias'] = create_bias_variable(
                        'postprocess2_bias',
                        [self.quantization_channels])
                var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        with tf.name_scope('causal_layer'):
            if self.velocity_input:
                weights_filter = self.variables['causal_layer']['filter_velocity']
                return causal_conv(input_batch, weights_filter, 1, velocity_input=True)
            else:
                weights_filter = self.variables['causal_layer']['filter']
                return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               global_condition_batch, output_width, minor_condition_batch=None):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             global_conditioning_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        variables = self.variables['dilated_stack'][layer_index]

        if self.velocity_input:
            weights_filter = variables['filter_velocity']
            weights_gate = variables['gate_velocity']
            #out_width
            conv_filter = causal_conv(input_batch, weights_filter, dilation, velocity_input=True)
            conv_gate = causal_conv(input_batch, weights_gate, dilation, velocity_input=True)
        else:
            weights_filter = variables['filter']
            weights_gate = variables['gate']
            conv_filter = causal_conv(input_batch, weights_filter, dilation)
            conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if global_condition_batch is not None:
            if self.velocity_input:
                raise NotImplementedError("Global_condition_batch only supports non-velocity case, \
                                            i.e. (bs, T, STH, channels) not supported.")
            #pad global_condition!!
            if self.load_chord:
                cut_width = tf.shape(conv_filter)[1]
                global_condition_batch = tf.slice(global_condition_batch, [0, 0, 0], [-1, cut_width,-1])
                if self.chain_mel and self.condition_restriction is not None:
                    minor_condition_batch = tf.slice(minor_condition_batch, [0, 0, 0], [-1, cut_width, -1])

            if self.chain_mel and self.condition_restriction is not None:
                if layer_index>=self.condition_restriction:
                    weights_gc_filter = variables['gc_filtweights']
                    weights_gc_gate = variables['gc_gateweights']
                    conv_filter = conv_filter + tf.nn.conv1d(global_condition_batch, weights_gc_filter,
                            stride=1, padding="SAME", name="gc_filter")
                    conv_gate = conv_gate + tf.nn.conv1d(global_condition_batch, weights_gc_gate,
                            stride=1, padding="SAME", name="gc_gate")
                else:
                    weights_gc_filter_minor = variables['minor_filtweights']
                    weights_gc_gate_minor = variables['minor_gateweights']
                    conv_filter = conv_filter +\
                            tf.nn.conv1d(minor_condition_batch, weights_gc_filter_minor,
                                    stride=1, padding="SAME", name="minor_filter")
                    conv_gate = conv_gate +\
                            tf.nn.conv1d(minor_condition_batch, weights_gc_gate_minor,
                                    stride=1, padding="SAME", name="minor_gate")
            elif self.condition_restriction is None or layer_index<self.condition_restriction:
                weights_gc_filter = variables['gc_filtweights']
                weights_gc_gate = variables['gc_gateweights']
                conv_filter = conv_filter + tf.nn.conv1d(global_condition_batch,
                                                    weights_gc_filter,
                                                    stride=1,
                                                    padding="SAME",
                                                    name="gc_filter")
                conv_gate = conv_gate + tf.nn.conv1d(global_condition_batch,
                                                 weights_gc_gate,
                                                 stride=1,
                                                 padding="SAME",
                                                 name="gc_gate")

        if self.use_biases:
            #TODO velocity, assume bias broadcast works.
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
        #out = tf.nn.relu(conv_filter) * tf.nn.relu(conv_gate)
        # The 1x1 conv to produce the residual output
        if self.velocity_input:
            weights_dense = variables['dense_velocity']
            transformed = tf.nn.conv2d(
                    out, weights_dense, strides=[1,1,1,1], padding="SAME", name="dense")
        else:
            weights_dense = variables['dense']
            transformed = tf.nn.conv1d(
                    out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        #TODO: BUG: HERE REQUIRES VERY LONG INPUT LIKE jazz_224
        if self.velocity_input:
            out_skip = tf.slice(out, [0, skip_cut, 0, 0], [-1, -1, -1, -1])
        else:
            out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        #out_skip_slice = tf.slice(out, [0, tf.abs(skip_cut), 0], [-1, -1, -1])
        #for velocity, outdated
        #out_skip_pad = tf.image.resize_image_with_crop_or_pad(out, tf.shape(out)[0], output_width)
        #out_skip = tf.cond(tf.less(skip_cut, 0),
                    #lambda: out_skip_pad,
                    #lambda: out_skip_slice)

        if self.velocity_input:
            weights_skip = variables['skip_velocity']
            skip_contribution = tf.nn.conv2d(
                    out_skip, weights_skip, strides=[1,1,1,1], padding="SAME", name="skip")
        else:
            weights_skip = variables['skip']
            skip_contribution = tf.nn.conv1d(
                    out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            #TODO: about velocity, again assuming broadcast works
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        if self.histograms:
            layer = 'layer{}'.format(layer_index)
            tf.summary.histogram(layer + '_filter', weights_filter)
            tf.summary.histogram(layer + '_gate', weights_gate)
            tf.summary.histogram(layer + '_dense', weights_dense)
            tf.summary.histogram(layer + '_skip', weights_skip)
            if self.use_biases:
                tf.summary.histogram(layer + '_biases_filter', filter_bias)
                tf.summary.histogram(layer + '_biases_gate', gate_bias)
                tf.summary.histogram(layer + '_biases_dense', dense_bias)
                tf.summary.histogram(layer + '_biases_skip', skip_bias)

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        if self.velocity_input:
            input_batch = tf.slice(input_batch, [0, input_cut, 0, 0], [-1, -1, -1, -1])
        else:
            input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed

    def _generator_conv(self, input_batch, state_batch, weights):
        '''Perform convolution for a single convolutional processing step.'''
        #TODO velocity assume works
        if self.velocity_input:
            past_weights = weights[0, :, :, :]
            curr_weights = weights[1, :, :, :]
        else:
            past_weights = weights[0, :, :]
            curr_weights = weights[1, :, :]

        output = tf.matmul(state_batch, past_weights) + tf.matmul(input_batch, curr_weights)
        return output

    def _generator_causal_layer(self, input_batch, state_batch):
        with tf.name_scope('causal_layer'):
            if self.velocity_input:
                weights_filter = self.variables['causal_layer']['filter_velocity']
            else:
                weights_filter = self.variables['causal_layer']['filter']
            output = self._generator_conv(
                input_batch, state_batch, weights_filter)
        return output

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
                                  dilation, global_condition_batch, minor_condition_batch=None):
        variables = self.variables['dilated_stack'][layer_index]

        if self.velocity_input:
            weights_filter = variables['filter_velocity']
            weights_gate = variables['gate_velocity']
            output_filter = self._generator_conv(
                    input_batch, state_batch, weights_filter)
            output_gate = self._generator_conv(
                    input_batch, state_batch, weights_filter)
        else:
            weights_filter = variables['filter']
            weights_gate = variables['gate']
            output_filter = self._generator_conv(
                input_batch, state_batch, weights_filter)
            output_gate = self._generator_conv(
                input_batch, state_batch, weights_gate)

        if global_condition_batch is not None:
            if self.velocity_input:
                raise NotImplementedError("Velocity not implemented for global.")
            if self.load_chord:
                #cut_width = tf.shape(output_filter)[1]
                #global_condition_batch = tf.slice(global_condition_batch, [0, 0, 0], [-1, cut_width, -1])
                if self.chain_mel and self.condition_restriction is not None:
                    #minor_condition_batch = tf.slice(minor_condition_batch, [0, 0, 0], [-1, cut_width, -1])
                    minor_condition_batch = tf.reshape(minor_condition_batch, shape=(1, -1))
                global_condition_batch = tf.reshape(global_condition_batch, shape=(1, -1))

            if self.chain_mel and self.condition_restriction is not None:
                if layer_index>=self.condition_restriction:
                    weights_gc_filter = variables['gc_filtweights']
                    weights_gc_filter = weights_gc_filter[0, :, :]
                    weights_gc_gate = variables['gc_gateweights']
                    weights_gc_gate = weights_gc_gate[0, :, :]
                    output_filter += tf.matmul(global_condition_batch, weights_gc_filter)
                    output_gate += tf.matmul(global_condition_batch, weights_gc_gate)
                else:
                    weights_gc_filter_minor = variables['minor_filtweights']
                    weights_gc_filter_minor = weights_gc_filter_minor[0, :, :]
                    weights_gc_gate_minor = variables['minor_gateweights']
                    weights_gc_gate_minor = weights_gc_gate_minor[0, :, :]
                    output_filter += tf.matmul(minor_condition_batch, weights_gc_filter_minor)
                    output_gate += tf.matmul(minor_condition_batch, weights_gc_gate_minor)
            elif self.condition_restriction is None or layer_index<self.condition_restriction:
                weights_gc_filter = variables['gc_filtweights']
                weights_gc_filter = weights_gc_filter[0, :, :]
                output_filter += tf.matmul(global_condition_batch,
                                       weights_gc_filter)
                weights_gc_gate = variables['gc_gateweights']
                weights_gc_gate = weights_gc_gate[0, :, :]
                output_gate += tf.matmul(global_condition_batch,
                                     weights_gc_gate)

        if self.use_biases:
            output_filter = output_filter + variables['filter_bias']
            output_gate = output_gate + variables['gate_bias']

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)
        #out = tf.nn.relu(output_filter) * tf.nn.relu(output_gate)

        if self.velocity_input:
            weights_dense = variables['dense_velocity']
            transformed = tf.matmul(out, weights_dense[0, :, :, :])
        else:
            weights_dense = variables['dense']
            transformed = tf.matmul(out, weights_dense[0, :, :])
        if self.use_biases:
            transformed = transformed + variables['dense_bias']

        if self.velocity_input:
            weights_skip = variables['skip_velocity']
            skip_contribution = tf.matmul(out, weights_skip[0, :, :, :])
        else:
            weights_skip = variables['skip']
            skip_contribution = tf.matmul(out, weights_skip[0, :, :])
        if self.use_biases:
            skip_contribution = skip_contribution + variables['skip_bias']

        return skip_contribution, input_batch + transformed

    def _create_network(self, input_batch, global_condition_batch, minor_condition_batch=None):
        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch

        # Pre-process the input with a regular convolution
        #if self.scalar_input:
            #initial_channels = 1
        #else:
            #initial_channels = self.quantization_channels

        current_layer = self._create_causal_layer(current_layer)

        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                        global_condition_batch, output_width,
                        minor_condition_batch=minor_condition_batch)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            if self.velocity_input:
                w1 = self.variables['postprocessing']['postprocess1_velocity']
                w2 = self.variables['postprocessing']['postprocess2_velocity']
            else:
                w1 = self.variables['postprocessing']['postprocess1']
                w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            if self.histograms:
                tf.summary.histogram('postprocess1_weights', w1)
                tf.summary.histogram('postprocess2_weights', w2)
                if self.use_biases:
                    tf.summary.histogram('postprocess1_biases', b1)
                    tf.summary.histogram('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            if self.velocity_input:
                conv1 = tf.nn.conv2d(transformed1, w1, strides=[1,1,1,1], padding="SAME")
            else:
                conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                #TODO velocity assume braodcast works
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            if self.velocity_input:
                conv2 = tf.nn.conv2d(transformed2, w2, strides=[1,1,1,1], padding="SAME")
            else:
                conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)

        return conv2

    def _create_generator(self, input_batch, global_condition_batch, minor_condition_batch=None):
        '''Construct an efficient incremental generator.'''
        init_ops = []
        push_ops = []
        outputs = []
        current_layer = input_batch

        #TODO: velocity, problem with FIFO HERE??
        if self.velocity_input:
            q = tf.FIFOQueue(1, dtypes=tf.float32,shapes=(self.batch_size, 2, self.quantization_channels))
            init = q.enqueue_many(tf.zeros((1, self.batch_size, 2, self.quantization_channels)))
        else:
            q = tf.FIFOQueue(
                1,
                dtypes=tf.float32,
                shapes=(self.batch_size, self.quantization_channels))
            init = q.enqueue_many(
                tf.zeros((1, self.batch_size, self.quantization_channels)))

        current_state = q.dequeue()
        push = q.enqueue([current_layer])
        init_ops.append(init)
        push_ops.append(push)

        current_layer = self._generator_causal_layer(
                            current_layer, current_state)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    if self.velocity_input:
                        q = tf.FIFOQueue(
                                dilation,
                                dtypes=tf.float32,
                                shapes=(self.batch_size, 2, self.residual_channels))
                        init = q.enqueue_many(
                                tf.zeros((dilation, self.batch_size, 2, self.residual_channels)))
                    else:
                        q = tf.FIFOQueue(
                            dilation,
                            dtypes=tf.float32,
                            shapes=(self.batch_size, self.residual_channels))
                        init = q.enqueue_many(
                            tf.zeros((dilation, self.batch_size,
                                      self.residual_channels)))

                    current_state = q.dequeue()
                    push = q.enqueue([current_layer])
                    init_ops.append(init)
                    push_ops.append(push)

                    output, current_layer = self._generator_dilation_layer(
                        current_layer, current_state, layer_index, dilation,
                        global_condition_batch, minor_condition_batch=minor_condition_batch)
                    outputs.append(output)
        self.init_ops = init_ops
        self.push_ops = push_ops

        with tf.name_scope('postprocessing'):
            variables = self.variables['postprocessing']
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            if self.velocity_input:
                w1 = variables['postprocess1_velocity']
                w2 = variables['postprocess2_velocity']
            else:
                w1 = variables['postprocess1']
                w2 = variables['postprocess2']

            if self.use_biases:
                b1 = variables['postprocess1_bias']
                b2 = variables['postprocess2_bias']

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)

            if self.velocity_input:
                conv1 = tf.matmul(transformed1, w1[0, :, :, :])
            else:
                conv1 = tf.matmul(transformed1, w1[0, :, :])
            if self.use_biases:
                conv1 = conv1 + b1
            transformed2 = tf.nn.relu(conv1)

            if self.velocity_input:
                conv2 = tf.matmul(transformed2, w2[0, :, :, :])
            else:
                conv2 = tf.matmul(transformed2, w2[0, :, :])
            if self.use_biases:
                conv2 = conv2 + b2
        return conv2

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.
        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            if self.velocity_input and not self.chain_vel:
                input_re = tf.reshape(input_batch, [self.batch_size, -1])
                encoded = tf.one_hot(input_re, depth=self.quantization_channels, dtype=tf.float32)
                shape = [self.batch_size, -1, 2, self.quantization_channels]
                encoded = tf.reshape(encoded, shape)
            else:
                encoded = tf.one_hot(
                    input_batch,
                    depth=self.quantization_channels,
                    dtype=tf.float32)
                shape = [self.batch_size, -1, self.quantization_channels]
                encoded = tf.reshape(encoded, shape)
        return encoded

    def _embed_gc(self, global_condition):
        '''Returns embedding for global condition.
        :param global_condition: Either ID of global condition for
               tf.nn.embedding_lookup or actual embedding. The latter is
               experimental.
        :return: Embedding or None
        '''
        embedding = None
        if self.chain_vel or self.chain_mel:
            #[bs, -1, 2]
            embedding_prod, embedding_cond = tf.split(global_condition, [1, 1], -1)
            if self.chain_vel:
                embedding_table_prod = self.variables['embeddings']['gc_vel']
            else:
                embedding_table_prod = self.variables['embeddings']['gc_pitch']
            embedding_table_cond = self.variables['embeddings']['gc_embedding']
            embedding_prod = tf.nn.embedding_lookup(embedding_table_prod, embedding_prod)
            embedding_cond = tf.nn.embedding_lookup(embedding_table_cond, embedding_cond)
            embedding = embedding_prod + embedding_cond
            embedding_prod = tf.reshape(
                    embedding_prod, [self.batch_size, -1, self.global_condition_channels])
            embedding_cond = tf.reshape(
                    embedding_cond, [self.batch_size, -1, self.global_condition_channels])
        elif self.global_condition_cardinality is not None:
            # Only lookup the embedding if the global condition is presented
            # as an integer of mutually-exclusive categories ...
            embedding_table = self.variables['embeddings']['gc_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               global_condition)
        elif global_condition is not None:
            # ... else the global_condition (if any) is already provided
            # as an embedding.
            # In this case, the number of global_embedding channels must be
            # equal to the the last dimension of the global_condition tensor.
            gc_batch_rank = len(global_condition.get_shape())
            dims_match = (global_condition.get_shape()[gc_batch_rank - 1] ==
                          self.global_condition_channels)
            if not dims_match:
                raise ValueError('Shape of global_condition {} does not'
                                 ' match global_condition_channels {}.'.
                                 format(global_condition.get_shape(),
                                        self.global_condition_channels))
            embedding = global_condition

        if self.chain_vel or self.chain_mel or self.load_chord:
            embedding = tf.reshape(
                    embedding, [self.batch_size, -1, self.global_condition_channels])
            if self.chain_mel and self.condition_restriction is not None:
                embedding_prod = tf.reshape(embedding_prod,
                    [self.batch_size, -1, self.global_condition_channels])
                embedding_cond = tf.reshape(embedding_cond,
                    [self.batch_size, -1, self.global_condition_channels])
        elif embedding is not None and not self.load_chord:
            embedding = tf.reshape(
                embedding,
                [self.batch_size, 1, self.global_condition_channels])

        if self.chain_mel and self.condition_restriction is not None:
            return embedding_prod, embedding_cond
        return embedding

    def _embed_lc(self, condition, network_input_width):
        if condition is None:
            raise ValueError('Condition batch in local condition layer can not be None')
        out_shape = [self.batch_size, network_input_width, 1 , self.local_condition_channels]
        embedding = tf.expand_dims(condition, 2)
        embedding = tf.expand_dims(embedding, 3)
        local_w = self.variables['embeddings']['lc_embedding']
        embedding = tf.nn.conv2d_transpose(embedding, local_w, output_shape=tf.stack(out_shape), strides = [1, self.local_upsample_rate, 1, 1], padding='SAME')
        embedding = tf.reshape(
                embedding, [self.batch_size, -1, self.local_condition_channels])
        return embedding

    def predict_proba(self, waveform, global_condition=None, name='wavenet'):
        '''Computes the probability distribution of the next sample based on
        all samples in the input waveform.
        If you want to generate audio by feeding the output of the network back
        as an input, see predict_proba_incremental for a faster alternative.
        Note that for non-fast generation, each time the output is feeded back as input'''
        with tf.name_scope(name):
            if self.scalar_input:
                #TODO: may not working since first dim
                encoded = tf.cast(waveform, tf.float32)
                encoded = tf.reshape(encoded, [-1, 1])
            elif self.velocity_input:
                encoded = self._one_hot(waveform)
                encoded = tf.reshape(encoded, [-1, 2, self.quantization_channels])
                encoded = tf.expand_dims(encoded, 0)
            else:
                encoded = self._one_hot(waveform)
            #INPUT??

            vel_embedding = None
            if self.local_condition_channels:
                gc_embedding = self._embed_lc(global_condition)
            elif self.chain_mel and self.condition_restriction is not None:
                vel_embedding, gc_embedding = self._embed_gc(global_condition)
            else:
                gc_embedding = self._embed_gc(global_condition)

            raw_output = self._create_network(encoded, gc_embedding, minor_condition_batch=vel_embedding)
            if self.velocity_input:
                out = tf.reshape(raw_output, [-1, 2, self.quantization_channels])
            else:
                out = tf.reshape(raw_output, [-1, self.quantization_channels])
            # Cast to float64 to avoid bug in TensorFlow

            if self.velocity_input:
                proba_melody, proba_velocity = tf.split(out, [1,1], 1)
                proba_melody = tf.reshape(proba_melody, [-1, self.quantization_channels])
                proba_velocity = tf.reshape(proba_velocity, [-1, self.quantization_channels])
                #TODO cast to tf.float64 instead?
                proba_melody = tf.cast(tf.nn.softmax(tf.cast(proba_melody, tf.float64)), tf.float32)
                proba_velocity = tf.cast(tf.nn.softmax(tf.cast(proba_velocity, tf.float64)), tf.float32)
                last_m = tf.slice(
                        proba_melody,
                        [tf.shape(proba_melody)[0] - 1, 0],
                        [1, self.quantization_channels])
                last_v = tf.slice(proba_velocity,
                        [tf.shape(proba_velocity)[0] -1, 0],
                        [1, self.quantization_channels])
                return tf.stack([tf.reshape(last_m, [-1]), tf.reshape(last_v, [-1])])
            else:
                proba = tf.cast(
                    tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
                last = tf.slice(
                    proba,
                    [tf.shape(proba)[0] - 1, 0],
                    [1, self.quantization_channels])
                return tf.reshape(last, [-1])

    def predict_proba_incremental(self, waveform, global_condition=None,
                                  name='wavenet'):
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''
        if self.filter_width > 2:
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")
        if self.scalar_input:
            raise NotImplementedError("Incremental generation does not "
                                      "support scalar input yet.")
        """waveform doesn't include padding"""
        with tf.name_scope(name):
            if self.velocity_input:
                #waveform: (1,2)
                encoded = self._one_hot(waveform)
                encoded = tf.reshape(encoded, [-1, 2, self.quantization_channels])
            else:
                encoded = tf.one_hot(waveform, self.quantization_channels, dtype=tf.float32)
                encoded = tf.reshape(encoded, [-1, self.quantization_channels])

            vel_embedding = None
            if self.local_condition_channels:
                gc_embedding = self._embed_lc(global_condition)
            elif self.chain_mel and self.condition_restriction is not None:
                vel_embedding, gc_embedding = self._embed_gc(global_condition)
            else:
                gc_embedding = self._embed_gc(global_condition)

            raw_output = self._create_generator(encoded, gc_embedding,
                                                minor_condition_batch=vel_embedding)
            if self.velocity_input:
                out = tf.reshape(raw_output, [-1, 2, self.quantization_channels])
            else:
                out = tf.reshape(raw_output, [-1, self.quantization_channels])

            if self.velocity_input:
                proba_melody, proba_velocity = tf.split(out, [1, 1], 1)
                proba_melody = tf.reshape(proba_melody, [-1, self.quantization_channels])
                proba_velocity = tf.reshape(proba_velocity, [-1, self.quantization_channels])
                proba_melody = tf.cast(tf.nn.softmax(tf.cast(proba_melody, tf.float64)), tf.float64)
                proba_velocity = tf.cast(tf.nn.softmax(tf.cast(proba_velocity, tf.float64)), tf.float64)
            else:
                proba = tf.cast(
                    tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)

            #TODO: ORDER OF SOFTMAX??
            if self.velocity_input:
                last_m = tf.slice(
                        proba_melody,
                        [tf.shape(proba_melody)[0] - 1, 0],
                        [1, self.quantization_channels])
                last_v = tf.slice(
                        proba_velocity,
                        [tf.shape(proba_velocity)[0] - 1, 0],
                        [1, self.quantization_channels])
            else:
                last = tf.slice(
                        proba,
                        [tf.shape(proba)[0] - 1, 0],
                        [1, self.quantization_channels])

            if self.velocity_input:
                #TODO velocity (2, 128)
                return tf.stack([tf.reshape(last_m, [-1]), tf.reshape(last_v, [-1])])
            else:
                return tf.reshape(last, [-1])

    def loss(self,
             input_batch,
             global_condition_batch=None,
             l2_regularization_strength=None,
             name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.name_scope(name):
            # We mu-law encode and quantize the input audioform.
            #TODO
            if self.midi_input:
                encoded_input = tf.to_int32(input_batch)
            else:
                encoded_input = mu_law_encode(input_batch, self.quantization_channels)

            network_input_width = tf.shape(encoded_input)[1] - 1
            if self.local_condition_channels and global_condition_batch is not None:
                gc_embedding = self._embed_lc(global_condition_batch, network_input_width)
            elif self.chain_mel and self.condition_restriction is not None:
                vel_embedding, gc_embedding = self._embed_gc(global_condition_batch)
            else:
                gc_embedding = self._embed_gc(global_condition_batch)

            encoded = self._one_hot(encoded_input)
            if self.scalar_input:
                network_input = tf.reshape(
                    tf.cast(input_batch, tf.float32),
                    [self.batch_size, -1, 1])
            else:
                network_input = encoded

            # Cut off the last sample of network input to preserve causality.
            network_input_width = tf.shape(network_input)[1] - 1
            if self.velocity_input:
                network_input = tf.slice(network_input, [0, 0, 0, 0],
                                        [-1, network_input_width, -1, -1])
            else:
                network_input = tf.slice(network_input, [0, 0, 0],
                                     [-1, network_input_width, -1])

            if gc_embedding:
                gc_embedding = tf.slice(gc_embedding, [0, 0, 0], [-1, network_input_width, -1])
            if self.chain_mel and self.condition_restriction is not None:
                vel_embedding = tf.slice(vel_embedding, [0, 0, 0], [-1, network_input_width, -1])
            else:
                vel_embedding = None
            raw_output = self._create_network(network_input, gc_embedding, minor_condition_batch=vel_embedding)

            #TODO velocity HERE!!
            with tf.name_scope('loss'):
                # Cut off the samples corresponding to the receptive field
                # for the first predicted sample.
                if self.velocity_input:
                    target_output = tf.slice(
                            tf.reshape(
                                encoded,
                                [self.batch_size, -1, 2, self.quantization_channels]),
                            [0, self.receptive_field, 0, 0],
                            [-1, -1, -1, -1])
                else:
                    target_output = tf.slice(
                            tf.reshape(
                                encoded,
                                [self.batch_size, -1, self.quantization_channels]),
                            [0, self.receptive_field, 0],
                            [-1, -1, -1])

                if self.velocity_input:
                    #TODO seperate velocity and melody
                    raw_output = tf.reshape(raw_output, [self.batch_size, -1, 2, self.quantization_channels])
                    target_output_list = tf.unstack(target_output, axis=2)
                    target_output_melody = target_output_list[0]
                    target_output_velocity = target_output_list[1]
                    prediction_list = tf.unstack(raw_output, axis=2)
                    prediction_melody = prediction_list[0]
                    prediction_velocity = prediction_list[1]
                else:
                    target_output = tf.reshape(target_output,
                                           [-1, self.quantization_channels])
                    prediction = tf.reshape(raw_output,
                                        [-1, self.quantization_channels])

                if self.velocity_input:
                    loss_melody = tf.nn.softmax_cross_entropy_with_logits(
                            logits=prediction_melody, labels=target_output_melody)
                    loss_velocity = tf.nn.softmax_cross_entropy_with_logits(
                            logits=prediction_velocity, labels=target_output_velocity)
                    reduced_loss = 0.999*tf.reduce_mean(loss_melody) + 0.001*tf.reduce_mean(loss_velocity)

                else:
                    loss = tf.nn.softmax_cross_entropy_with_logits(
                            logits=prediction,
                            labels=target_output)
                    reduced_loss = tf.reduce_mean(loss)

                tf.summary.scalar('loss', reduced_loss)

                if l2_regularization_strength is None:
                    return reduced_loss
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss +
                                  l2_regularization_strength * l2_loss)

                    tf.summary.scalar('l2_loss', l2_loss)
                    tf.summary.scalar('total_loss', total_loss)

                    return total_loss

