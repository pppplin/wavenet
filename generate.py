from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
#import os

import librosa
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, audio_reader
from wavenet.audio_reader import load_audio_velocity
from reverse_pianoroll import array_to_pretty_midi, array_to_pretty_midi_velocity

from gpu import define_gpu
import pretty_midi
define_gpu(2)

SAMPLES = 16000
TEMPERATURE = 1.0
LOGDIR = './logdir/train/2017-11-03T10-40-54/model.ckpt-1400'
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None
SILENCE_THRESHOLD = 0 #0.1


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
    parser.add_argument('--load_velocity', type=_str_to_bool, default=False, help='Whether to include velocity in training.')
    parser.add_argument('--midi_input', type=_str_to_bool, default=True)
    parser.add_argument('--load_chord', type=_str_to_bool, default=False, help='Whether to include chord in training.(midi only)')
    arguments = parser.parse_args()
    if arguments.gc_channels is not None:
        if arguments.gc_cardinality is None:
            raise ValueError("Globally conditioning but gc_cardinality not "
                             "specified. Use --gc_cardinality=377 for full "
                             "VCTK corpus.")

        if arguments.gc_id is None and not arguments.load_chord:
            raise ValueError("Globally conditioning, but global condition was "
                              "not specified. Use --gc_id to specify global "
                              "condition.")

    return arguments


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))

def convert_to_mid(array, sample_rate, filename, velocity, op):
    """
    array: np.ndarray, shape = (?, 1), directed generated array
    save as midi file
    """
    arr = np.asarray(array)
    if velocity:
        mid = array_to_pretty_midi_velocity(arr, fs = sample_rate, op=op)
        #mid_mel = array_to_pretty_midi(arr[:, 0], fs = sample_rate, op=op)
        #temp_ls = filename.split('/')
        #mel_file_name = ''
        #for i in range(len(temp_ls)):
        #    if (i==(len(temp_ls)-1)):
        #        mel_file_name += 'mel_'
        #        mel_file_name += temp_ls[i]
        #    elif (i==(len(temp_ls)-2)):
        #        mel_file_name += temp_ls[i]
        #        mel_file_name += '_mel_only/'
        #    else:
        #        mel_file_name += temp_ls[i]
        #        mel_file_name += "/"
        #mid_mel.write(mel_file_name)
    else:
        mid = array_to_pretty_midi(arr, fs = sample_rate, op = op)
    mid.write(filename)

def create_seed(filename,
                sample_rate,
                quantization_channels,
                window_size,
                silence_threshold=SILENCE_THRESHOLD):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio_reader.trim_silence(audio, silence_threshold)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))

    return quantized[:cut_index]

def create_midi_seed(filename,
                     samples_num,
                     sample_rate,
                     quantization_channels,
                     window_size,
                     silence_threshold=SILENCE_THRESHOLD, velocity = False, chord = False):
    midi = pretty_midi.PrettyMIDI(filename)
    midi.instruments = [midi.instruments[0]]
    midi = midi.get_piano_roll(fs = sample_rate, times = None)

    midi = np.swapaxes(midi, 0, 1)
    midi = np.argmax(midi, axis = -1)
    #TODO
    midi = midi.astype(np.float32)
    cut_index = np.size(midi) if np.size(midi)<window_size else window_size
    if velocity:
        midi = np.reshape(midi, (-1, 1))
        velocity = load_audio_velocity(filename, sample_rate)
        midi = np.concatenate((midi, velocity), axis=1)
    #quantized = mu_law_encode(midi, quantization_channels)
    #cut_index = np.size(midi) if np.size(midi)<tf.constant(window_size) else tf.constant(window_size)
    #cut_index = tf.cond(tf.constant(midi.size) < tf.constant(window_size),
    #                    lambda: tf.constant(midi.size),
    #                    lambda: tf.constant(window_size))
    if chord:
        cmidi = pretty_midi.PrettyMIDI(filename)
        cmidi.instruments = [cmidi.instruments[-1]]
        chords = cmidi.get_piano_roll(fs = sample_rate, times = None)
        chords = np.swapaxes(chords, 0, 1)
        chords = np.argmax(chords, axis=-1)
        chords_0 = chords[:midi.shape[0]]
        chords_1 = chords[cut_index: cut_index+samples_num]
        midi = np.reshape(midi, (-1, 1))
        chords_0 = np.reshape(chords_0, (-1, 1))
        chords_1 = np.reshape(chords_1, (-1, 1))
        midi = np.concatenate((midi, chords_0), axis = 1)
        return tf.stack(midi[:cut_index]), chords_1
    return tf.stack(midi[:cut_index])


def main():
    args = get_arguments()
    #started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    #logdir = os.path.join(args.logdir, 'generate', started_datestring)
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
        velocity_input=args.load_velocity,
        midi_input=wavenet_params["midi_input"],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'],
        load_chord=args.load_chord,
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=args.gc_cardinality)

    samples = tf.placeholder(tf.int32)

    if args.load_chord:
        #use fast generation as default
        chords = tf.placeholder(tf.int32)
        next_sample = net.predict_proba_incremental(samples, chords)
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
        elif args.load_chord:
            seed, chords_cont = create_midi_seed(args.wav_seed,
                                        args.samples,
                                        wavenet_params['sample_rate'],
                                        quantization_channels,
                                        net.receptive_field,
                                        velocity = args.load_velocity,
                                        chord = args.load_chord)
        else:
            seed = create_midi_seed(args.wav_seed,
                            args.samples,
                            wavenet_params['sample_rate'],
                            quantization_channels,
                            net.receptive_field,
                            velocity = args.load_velocity,
                            chord = args.load_chord)

        waveform = sess.run(seed).tolist()
        if args.load_velocity:
            wave_array = np.asarray(waveform)
            sample_max = np.max(wave_array[:, 0])
            sample_min = np.min(wave_array[:, 0])
            velocity_max = np.max(wave_array[:, 1])
            velocity_min = np.min(wave_array[:, 1])
        elif args.load_chord:
            wave_array = np.asarray(waveform)
            sample_max = np.max(wave_array[:, 0])
            sample_min = np.min(wave_array[:, 0])
        else:
            sample_max = max(waveform)
            sample_min = min(waveform)
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

            if i % 100 == 0:
                print('Priming sample {}'.format(i))
            if args.load_chord:
                #load chord from waveform
                sess.run(outputs, feed_dict={samples: x[:, 0], chords: x[:, 1]})
            else:
                sess.run(outputs, feed_dict={samples: x})
        print('Done.')

    last_sample_timestamp = datetime.now()
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
        if args.load_chord:
            prediction = sess.run(outputs, feed_dict={samples: window[0], chords: window[1]})[0]
        else:
            prediction = sess.run(outputs, feed_dict={samples: window})[0]

        # Velocity outputs get some negligible error,  renormalized!
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

        if args.wav_seed is None:
            waveform.append(sample)
        elif args.wav_seed and not args.load_velocity and not args.load_chord and sample>=sample_min and sample<=sample_max:
            waveform.append(sample)
        elif args.wav_seed and args.load_chord and sample>=sample_min and sample<=sample_max:
            waveform.append([sample, chords_cont[step]])
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
            assert False
            #TODO: not implemented for velocity and chords
            out = sess.run(decode, feed_dict={samples: waveform})
            convert_to_mid(out, sample_rate, args.mid_out_path, args.load_velocity)
    # Introduce a newline to clear the carriage return from the progress.
    print()

    # Save the result as an audio summary.
    #datestring = str(datetime.now()).replace(' ', 'T')
    #writer = tf.summary.FileWriter(logdir)
    #tf.summary.audio('generated', decode, wavenet_params['sample_rate'])
    #summaries = tf.summary.merge_all()
    #summary_out = sess.run(summaries,
    #                       feed_dict={samples: np.reshape(waveform, [-1, 1])})
    #writer.add_summary(summary_out)

    # Save the result as file.
    #TODO
    #waveform = np.reshape(waveform, [-1, 1])
    if args.mid_out_path:
        #out = sess.run(decode, feed_dict={samples: waveform})
        #TODO
        #out = sess.run(decode, feed_dict={samples: waveform})
        #print(tf.shape(decode))
        if args.load_velocity:
            out = np.reshape(waveform, (-1, 2))
        elif args.load_chord:
            out = np.reshape(waveform, (-1, 2))
            out = np.reshape(out[:, 0], (-1, 1))
        else:
            out = np.reshape(waveform, (-1, 1))
        convert_to_mid(out, sample_rate, args.mid_out_path, args.load_velocity, args.wav_seed)

    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    main()
