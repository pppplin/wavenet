from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, mu_law_decode
from utils import create_midi_seed, create_seed, convert_to_mid
#from scipy.stats import entropy

from gpu import define_gpu
define_gpu(2)

SAMPLES = 16000
TEMPERATURE = 1.0
LOGDIR = './logdir/train/2017-11-03T10-40-54/model.ckpt-1400'
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None
SILENCE_THRESHOLD = 0 #0.1
VEL_SET = [0, 80, 95, 105]

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
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many samples to generate')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    #TODO
    parser.add_argument(
        '--mid_out_path',
        type=str,
        default=None,
        help='Path to output mid file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=False, #TODO: non-fast not working for non-velocity(probably because fitst dim), and fast not working for velocity
        help='Use fast generation, i.e. only feed in the last sample for prediction')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    parser.add_argument(
        '--gc_channels',
        type=int,
        default=None,
        help='Number of global condition embedding channels. Omit if no '
             'global conditioning.')
    parser.add_argument(
        '--gc_cardinality',
        type=int,
        default=None,
        help='Number of categories upon which we globally condition.')
    parser.add_argument(
        '--gc_id',
        type=int,
        default=None,
        help='ID of category to generate, if globally conditioned.')
    parser.add_argument('--condition_restriction', type=int, default=None, help='Provided manually, for chord conditioning')
    parser.add_argument('--load_velocity', type=_str_to_bool, default=False, help='Whether to include velocity in training.')
    parser.add_argument('--midi_input', type=_str_to_bool, default=True)
    parser.add_argument('--load_chord', type=_str_to_bool, default=False, help='Whether to include chord in training.(midi only)')
    parser.add_argument('--chain_mel', type=_str_to_bool, default=False, help='Chain melody.')
    parser.add_argument('--chain_vel', type=_str_to_bool, default=False, help='Chain velocity.')
    parser.add_argument('--init_chain', type=_str_to_bool, default=False, help='Initialize chain: all vel to 100, produce mel')
    parser.add_argument('--lc_channels', type=int, default=None, help='Number of local condition channels.')
    parser.add_argument('--num_iterations', type=int, default=None, help='Number of iterations (for chain).')
    arguments = parser.parse_args()
    if arguments.gc_channels is not None:
        if arguments.gc_cardinality is None:
            raise ValueError("Globally conditioning but gc_cardinality not "
                             "specified. Use --gc_cardinality=377 for full "
                             "VCTK corpus.")

        if arguments.gc_id is None and (not (arguments.load_chord or arguments.chain_mel or arguments.chain_vel)):
            raise ValueError("Globally conditioning, but global condition was "
                              "not specified. Use --gc_id to specify global "
                              "condition.")

    return arguments

def main():
    args = get_arguments()
    #started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    #logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()
    #local_upsample_rate = wavenet_params["sample_rate"]/wavenet_params["local_sample_rate"]
    local_upsample_rate = None
    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        velocity_input=args.load_velocity,
        midi_input=wavenet_params["midi_input"],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'],
        load_chord=args.load_chord,
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=args.gc_cardinality,
        local_condition_channels=args.lc_channels,
        local_upsample_rate=local_upsample_rate,
        condition_restriction=args.condition_restriction,
        chain_mel=args.chain_mel,
        chain_vel=args.chain_vel)

    """Restore model"""
    samples = tf.placeholder(tf.int32)
    if args.load_chord or args.chain_mel or args.chain_vel:
        #use fast generation as default
        global_cond = tf.placeholder(tf.int32)
        next_sample = net.predict_proba_incremental(samples, global_cond)
    elif args.fast_generation:
        next_sample = net.predict_proba_incremental(samples, args.gc_id)
    else:
        next_sample = net.predict_proba(samples, args.gc_id)

    if args.fast_generation:
        sess.run(tf.global_variables_initializer())
        sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)
    print('Done')

    """Import seed"""
    if not args.midi_input:
        decode = mu_law_decode(samples, wavenet_params['quantization_channels'])
    else:
        decode = tf.to_int32(samples)
    quantization_channels = wavenet_params['quantization_channels']
    sample_rate = wavenet_params['sample_rate']
    if args.wav_seed:
        if not args.midi_input:
            seed = create_seed(args.wav_seed,
                            wavenet_params['sample_rate'],
                            quantization_channels,
                            net.receptive_field)
        elif args.load_chord or args.chain_mel or args.chain_vel:
            seed, cond_cont = create_midi_seed(
                    filename=args.wav_seed,
                    samples_num=args.samples,
                    sample_rate=wavenet_params['sample_rate'],
                    window_size=net.receptive_field,
                    use_velocity=args.load_velocity,
                    use_chord=args.load_chord, chain_mel=args.chain_mel,
                    chain_vel=args.chain_vel, init=args.init_chain)
        elif local_upsample_rate:
            raise ValueError("Not implemented yet")
            #TODO add a local variable, dont use sample rate
        else:
            #raise ValueError("Not implemented yet")
            seed = create_midi_seed(args.wav_seed,
                            samples_num=args.samples,
                            sample_rate=wavenet_params['sample_rate'],
                            window_size=net.receptive_field,
                            use_velocity = False,
                            use_chord = False)

        waveform = sess.run(seed).tolist()
        print(waveform)
        if args.load_velocity:
            wave_array = np.asarray(waveform)
            sample_max = np.max(wave_array[:, 0])
            temp_arr = wave_array[:, 0]
            sample_min = np.min(temp_arr[np.nonzero(temp_arr)])
            velocity_max = np.max(wave_array[:, 1])
            temp_arr = wave_array[:, 1]
            velocity_min = np.min(temp_arr[np.nonzero(temp_arr)])
        elif args.load_chord or args.chain_mel or args.chain_vel:
            wave_array = np.asarray(waveform)
            sample_max = np.max(wave_array[:, 0])
            temp_arr = wave_array[:, 0]
            sample_min = np.min(temp_arr[np.nonzero(temp_arr)])
        else:
            sample_max = max(waveform)
            temp_arr = np.asarray(waveform)
            sample_min = np.min(temp_arr[np.nonzero(temp_arr)])
    else:
        # Silence with a single random sample at the end.
        waveform = [quantization_channels / 2] * (net.receptive_field - 1)
        waveform.append(np.random.randint(quantization_channels))

    if args.fast_generation and args.wav_seed:
        # When using the incremental generation, we need to
        # feed in all priming samples one by one before starting the
        # actual generation.
        # TODO This could be done much more efficiently by passing the waveform
        # to the incremental generator as an optional argument, which would be
        # used to fill the queues initially.
        outputs = [next_sample]
        outputs.extend(net.push_ops)
        print('Priming generation...')
        for i, x in enumerate(waveform[-net.receptive_field: -1]):
            if args.load_velocity or args.load_chord:
                x = np.asarray(x)
                x = np.reshape(x, (-1, 2))
            elif args.chain_mel or args.chain_vel:
                x = np.asarray(x)
                x = np.reshape(x, (-1, 3))
            else:
                x = np.asarray(x)
                x = np.reshape(x, (-1, 1))
            if i % 100 == 0:
                print('Priming sample {}'.format(i))
            if args.chain_mel or args.chain_vel:
                #load global condition from waveform
                sess.run(outputs, feed_dict={samples: x[:, 0], global_cond: x[:, 1:]})
            elif args.load_chord:
                sess.run(outputs, feed_dict={samples: x[:, 0], global_cond: x[:, 1]})
            else:
                sess.run(outputs, feed_dict={samples: x[:, 0]})
        print('Done.')

    last_sample_timestamp = datetime.now()
    #TODO chain
    """Generation"""
    for step in range(args.samples):
        if args.fast_generation:
            outputs = [next_sample]
            outputs.extend(net.push_ops)#TODO: problem VELOCITY??
            window = waveform[-1]
        else:
            if len(waveform) > net.receptive_field:
                window = waveform[-net.receptive_field:]
            else:
                window = waveform
            outputs = [next_sample]

        # Run the WaveNet to predict the next sample.
        if args.chain_mel or args.chain_vel:
            prediction = sess.run(outputs, feed_dict={samples: window[0], global_cond: window[1:]})[0]
        elif args.load_chord:
            prediction = sess.run(outputs, feed_dict={samples: window[0], global_cond: window[1]})[0]
        else:
            prediction = sess.run(outputs, feed_dict={samples: window})[0]

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
        if args.temperature == 1.0 and not args.load_velocity:
            np.testing.assert_allclose(
                    prediction, scaled_prediction, atol=1e-5,
                    err_msg='Prediction scaling at temperature=1.0 '
                            'is not working as intended.')
        if args.load_velocity:
            sample_melody = np.random.choice(np.arange(quantization_channels), p=prediction[0])
            sample_velocity = np.random.choice(np.arange(quantization_channels), p=prediction[1])
        else:
            sample = np.random.choice(np.arange(quantization_channels), p=scaled_prediction)

        sample = sample_max if sample>sample_max else sample
        if sample<sample_min:
            if sample<(sample_min/2):
                sample = 0
            else:
                sample = sample_min
        if not (args.load_velocity or args.load_chord or args.chain_mel or args.chain_vel):
            waveform.append(sample)
        elif args.wav_seed and (args.chain_mel or args.chain_vel):
            waveform.append([sample, cond_cont[step][0], cond_cont[step][1]])
        elif args.wav_seed and args.load_chord:
            waveform.append([sample, cond_cont[step][0]])
        elif args.load_velocity:
            #TODO ONLY FOR TESTING
            if sample_melody>=sample_min and sample_melody<=sample_max \
                    and sample_velocity>=velocity_min and sample_velocity<=velocity_max:
                print(sample_melody, sample_velocity)
                waveform.append([sample_melody, sample_velocity])

        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

        # If we have partial writing, save the result so far.
        #TODO
        if (args.mid_out_path and args.save_every and
                (step + 1) % args.save_every == 0):
            raise ValueError("Not implemented for velocity, chord and chain.")
            #TODO: not implemented for velocity and chords
            out = sess.run(decode, feed_dict={samples: waveform})
            convert_to_mid(out, sample_rate, args.mid_out_path, args.load_velocity)
    # Introduce a newline to clear the carriage return from the progress.
    print()
    """Write to file"""
    if args.mid_out_path:
        if args.load_velocity:
            out = np.reshape(waveform, (-1, 2))
        elif args.load_chord:
            out = np.reshape(waveform, (-1, 2))
            out = np.reshape(out[:, 0], (-1, 1))
            print(out[-100:, :])
        elif args.chain_mel or args.chain_vel:
            out = np.reshape(waveform, (-1, 3))[:, :2]
        else:
            out = np.reshape(waveform, (-1, 1))
        convert_to_mid(array=out, sample_rate=wavenet_params['sample_rate'],
                filename=args.mid_out_path, op=args.wav_seed,
                velocity=args.load_velocity,
                chain_mel=args.chain_mel, chain_vel=args.chain_vel)
    print('Finished generating. The result can be viewed in TensorBoard.')

if __name__ == '__main__':
    main()
