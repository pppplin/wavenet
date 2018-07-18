from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel
from utils import create_midi_seed, convert_to_mid
#from scipy.stats import entropy

from gpu import define_gpu
define_gpu(2)

SAMPLES = 16000
TEMPERATURE = 1.0
WAVENET_PARAMS = './wavenet_params.json'
GC_CARDINALITY = 52
GC_CHANNELS = 64
CHECKPOINT_MEL = './logdir/Nottingham/train/2018-07-17T00-18-54/model.ckpt-3999'
CHECKPOINT_VEL = './logdir/Nottingham/train/2018-07-16T23-55-21/model.ckpt-600'

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument('--checkpoint_mel', type=CHECKPOINT_MEL, help='Chain melody checkpoint to generate from')
    parser.add_argument('--checkpoint_vel', type=CHECKPOINT_VEL, help='Chain velocity checkpoint to generate from')
    parser.add_argument('--samples', type=int, default=SAMPLES, help='How many samples to generate')
    parser.add_argument('--temperature', type=_ensure_positive_float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS, help='JSON file with the network parameters')
    parser.add_argument('--mid_out_path', type=str, default=None, help='Path to output mid file')
    parser.add_argument('--wav_seed', type=str, default=None, help='The wav file to start generation from')
    parser.add_argument('--gc_channels', type=int, default=GC_CHANNELS, help='Number of global condition embedding channels')
    parser.add_argument('--gc_cardinality', type=int, default=GC_CARDINALITY, help='Number of categories upon which we globally condition.')
    parser.add_argument('--init_chain', type=_str_to_bool, default=False, help='Initialize chain: all vel to 100, produce mel')
    parser.add_argument('--chain_mel', type=_str_to_bool, default=False, help='Chain melody.')
    parser.add_argument('--chain_vel', type=_str_to_bool, default=False, help='Chain velocity.')
    parser.add_argument('--num_iterations', type=int, default=None, help='Number of iterations (for chain).')

    arguments = parser.parse_args()
    return arguments

def generate_func(net, sess, saver, args, samples, next_sample, wavenet_params, global_cond, local_upsample_rate=None):
    global idx, pre_pred
    if idx%2==0:
        print('Restoring model from {}'.format(args.checkpoint_mel))
        if idx==0:
            args.init_chain=True
        #mel
    else:
        #vel
    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)
    print('Done')
    #TODO

    """idx=0 init vel all 90 produce mel, odd: vel, even: mel"""
    global idx, pre_pred
    """Import seed"""
    quantization_channels = wavenet_params['quantization_channels']
    sample_rate = wavenet_params['sample_rate']
    seed, cond_cont = create_midi_seed(args.wav_seed, args.samples,
            wavenet_params['sample_rate'], quantization_channels, net.receptive_field,
            chain_mel=args.chain_mel, chain_vel=args.chain_vel, init=args.init_chain)

    waveform = sess.run(seed).tolist()
    wave_array = np.asarray(waveform)
    sample_max = np.max(wave_array[:, 0])
    sample_min = np.min(wave_array[:, 0])

    outputs = [next_sample]
    outputs.extend(net.push_ops)

    print('Priming generation...')
    for i, x in enumerate(waveform[-net.receptive_field: -1]):
        x = np.asarray(x)
        x = np.reshape(x, (-1, 3))
        if i % 100 == 0:
            print('Priming sample {}'.format(i))
        sess.run(outputs, feed_dict={samples: x[:, 0], global_cond: x[:, 1:]})
        args.samples = i
        print('Done.')

    last_sample_timestamp = datetime.now()
    #TODO chain
    """Generation"""
    for step in range(args.samples):
        outputs = [next_sample]
        outputs.extend(net.push_ops)#TODO: problem VELOCITY??
        window = waveform[-1]

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: window[0], global_cond: window[1:]})[0]

        '''Velocity outputs get some negligible error, renormalized!'''
        #TODO velocity assume np.log and else working
        # Scale prediction distribution using temperature. If temperature==1, scale_prediction==prediction.
        np.seterr(divide='ignore')
        scaled_prediction = np.log(prediction) / args.temperature
        scaled_prediction = (scaled_prediction -
                             np.logaddexp.reduce(scaled_prediction))
        scaled_prediction = np.exp(scaled_prediction)
        np.seterr(divide='warn')
        # Prediction distribution at temperature=1.0 should be unchanged after
        # scaling.
        #TODO: velocity testing, SKIP FOR NOW!!
        if args.temperature == 1.0:
            np.testing.assert_allclose(
                    prediction, scaled_prediction, atol=1e-5,
                    err_msg='Prediction scaling at temperature=1.0 '
                            'is not working as intended.')
        sample = np.random.choice(np.arange(quantization_channels), p=scaled_prediction)

        sample = sample_max if sample>sample_max else sample
        sample = sample_min if sample<sample_min else sample
        waveform.append([sample, cond_cont[step][0], cond_cont[step][1]])

        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

    print()
    """Write to file"""
    if args.mid_out_path:
        out = np.reshape(waveform, (-1, 3))[:, :2]
        convert_to_mid(out, sample_rate, args.mid_out_path, args.wav_seed,
                args.load_velocity, args.chain_mel, args.chain_vel)
    print('Finished generating. The result can be viewed in TensorBoard.')

def main():
    #pre_pred = None
    args = get_arguments()
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()
    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        midi_input=wavenet_params["midi_input"],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'],
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=args.gc_cardinality,
        chain_mel=args.chain_mel,
        chain_vel=args.chain_vel)

    """Restore model"""
    samples = tf.placeholder(tf.int32)
    global_cond = tf.placeholder(tf.int32)
    next_sample = net.predict_proba_incremental(samples, global_cond)

    sess.run(tf.global_variables_initializer())
    sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    for idx in range(args.num_iterations):
        generate_func(net, sess, saver, args, samples, next_sample, wavenet_params, global_cond)

if __name__ == '__main__':
    main()
