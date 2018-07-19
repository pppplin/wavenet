from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import numpy as np
import tensorflow as tf
import pretty_midi
#import re

from wavenet import WaveNetModel
from utils import convert_to_mid, load_audio_velocity
from scipy.stats import entropy

from gpu import define_gpu
define_gpu(2)

SAMPLES = 16000 #Overwrite in create_midi_seed
TEMPERATURE = 1.0
WAVENET_PARAMS = './wavenet_params.json'
GC_CARDINALITY = 52
GC_CHANNELS = 64
CHECKPOINT_MEL = './logdir/Nottingham/train/2018-07-17T00-18-54/model.ckpt-3999'
CHECKPOINT_VEL = './logdir/Nottingham/train/2018-07-16T23-55-21/model.ckpt-600'
WAV_SEED = './Nottingham_melody_64_hashed/test_cut_melody/hpps_simple_chords_11.mid'
MID_OUT_PATH = './Nottingham_melody_64_hashed/generated/test_global-hpps_11.mid'

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
    parser.add_argument('--wav_seed', type=str, default=WAV_SEED, help='The wav file to start generation from')
    parser.add_argument('--checkpoint_mel', type=str, default=CHECKPOINT_MEL, help='Chain melody checkpoint to generate from')
    parser.add_argument('--checkpoint_vel', type=str, default=CHECKPOINT_VEL, help='Chain velocity checkpoint to generate from')
    parser.add_argument('--temperature', type=_ensure_positive_float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS, help='JSON file with the network parameters')
    parser.add_argument('--mid_out_path', type=str, default=MID_OUT_PATH, help='Path to output mid file')
    parser.add_argument('--gc_channels', type=int, default=GC_CHANNELS, help='Number of global condition embedding channels')
    parser.add_argument('--gc_cardinality', type=int, default=GC_CARDINALITY, help='Number of categories upon which we globally condition.')
    parser.add_argument('--num_iterations', type=int, default=None, help='Number of iterations (for chain).')

    arguments = parser.parse_args()
    return arguments

def create_midi_seed(filename, sample_rate, window_size, chain_mel=False, chain_vel=False, init=False):
    pm = pretty_midi.PrettyMIDI(filename)
    mel_instrument = [pm.instruments[0]]
    chord_instrument = [pm.instruments[2]]
    if chain_vel:
        cont_instrument = [pm.instruments[-1]]
    pm.instruments = mel_instrument
    midi = pm.get_piano_roll(fs=sample_rate, times=None)
    midi = np.swapaxes(midi, 0, 1)
    midi = np.argmax(midi, axis=-1)
    midi = np.reshape(midi, (-1, 1))
    midi = midi.astype(np.float32)
    last = True if chain_mel and not init else False
    velocity = load_audio_velocity(filename, sample_rate, last=last)
    velocity = np.reshape(velocity, (-1, 1))
    pm.instruments = chord_instrument
    chords = pm.get_piano_roll(fs=sample_rate, times=None)
    chords = np.swapaxes(chords, 0, 1)
    chords = np.argmax(chords, axis=-1)
    chords = np.reshape(chords, (-1, 1))
    if chain_mel or chain_vel:
        samples_num = np.size(chords) - np.size(midi)
    midi_size = np.size(midi)
    if init:
        vel_init = np.asarray([90]*(np.size(chords)-midi_size))
        vel_init = np.reshape(vel_init, (-1, 1))
        velocity = np.concatenate((velocity, vel_init), axis=0)
    #chords = chords[:min_len]
    cut_index = np.size(midi) if np.size(midi)<window_size else window_size
    if chain_mel:
        prod = np.concatenate((midi, velocity[:midi_size], chords[:midi_size]), axis=1)
        cont_cut_index = min(np.size(chords), np.size(velocity))
        cond = np.concatenate((velocity[:cont_cut_index], chords[:cont_cut_index]), axis=1)
        return prod[:cut_index], cond[cut_index: cut_index+samples_num, :], samples_num
    if chain_vel:
        pm.instruments = cont_instrument
        cont_midi = pm.get_piano_roll(fs=sample_rate, times=None)
        cont_midi = np.swapaxes(cont_midi, 0, 1)
        cont_midi = np.argmax(cont_midi, axis=-1)
        cont_midi = np.reshape(cont_midi, (-1, 1))
        cont_cut_index = min(np.size(chords), np.size(cont_midi))
        prod = np.concatenate((velocity[:midi_size], midi, chords[:midi_size]), axis=1)
        cond = np.concatenate((cont_midi[:cont_cut_index], chords[:cont_cut_index]), axis=1)
        return prod[:cut_index], cond[cut_index: cut_index+samples_num, :], samples_num

def batch_entropy(pk, qk):
    '''pk, qk are list of ndarrays, each is a probability distribution'''
    cut_len = min(len(pk), len(qk))
    pk = pk[:cut_len]
    qk = qk[:cut_len]
    kl_value = 0
    for i in range(cut_len):
        kl_value += entropy(pk=pk[i].tolist(), qk=qk[i].tolist())
    kl_value = kl_value/cut_len
    return kl_value

def generate_func(args, wavenet_params, chain_mel, chain_vel, init_chain):
    global idx, pre_pred, kl_result, variables_to_restore
    cur_pred = []

    """Define Session"""
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
        chain_mel=chain_mel,
        chain_vel=chain_vel)

    samples = tf.placeholder(tf.int32)
    global_cond = tf.placeholder(tf.int32)
    next_sample = net.predict_proba_incremental(samples, global_cond) #variables created here
    sess.run(tf.global_variables_initializer())
    sess.run(net.init_ops)

    #working tf.global_variables() increasing with script all the time!
    if init_chain:
        variables_to_restore = {
            var.name[:-2]: var for var in tf.global_variables()
            if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)
    print('Restoring model from {}'.format(ckpt))
    saver.restore(sess, ckpt)
    print('Done')

    """Import seed"""
    quantization_channels = wavenet_params['quantization_channels']
    sample_rate = wavenet_params['sample_rate']
    receptive_field = net.receptive_field
    if init_chain:
        feed_path = args.wav_seed
    else:
        feed_path = args.mid_out_path
    seed, cond_cont, samples_num = create_midi_seed(feed_path, sample_rate,
            receptive_field, chain_mel=chain_mel, chain_vel=chain_vel, init=init_chain)

    waveform = seed.tolist()
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
    print('Done.')

    last_sample_timestamp = datetime.now()
    """Generation"""
    for step in range(samples_num):
        outputs = [next_sample]
        outputs.extend(net.push_ops)
        window = waveform[-1]

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: window[0], global_cond: window[1:]})[0]

        '''Velocity outputs get some negligible error, renormalized!'''
        #TODO velocity assume np.log and else working
        # Scale prediction distribution using temperature. If temperature==1, scale_prediction==prediction.
        np.seterr(divide='ignore')
        scaled_prediction = np.log(prediction) / args.temperature
        scaled_prediction = (scaled_prediction - np.logaddexp.reduce(scaled_prediction))
        scaled_prediction = np.exp(scaled_prediction)
        if init_chain:
            pre_pred.append(scaled_prediction)
        elif chain_mel:
            cur_pred.append(scaled_prediction)
        np.seterr(divide='warn')
        # Prediction distribution at temperature=1.0 should be unchanged after
        # scaling.
        #TODO: velocity testing, SKIP FOR NOW!!
        if args.temperature == 1.0:
            np.testing.assert_allclose(prediction, scaled_prediction, atol=1e-5,
                    err_msg='Prediction scaling at temperature=1.0 is not working as intended.')
        sample = np.random.choice(np.arange(quantization_channels), p=scaled_prediction)

        sample = sample_max if sample>sample_max else sample
        sample = sample_min if sample<sample_min else sample
        try:
            waveform.append([sample, cond_cont[step][0], cond_cont[step][1]])
        except:
            print(step)
        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Sample {:3<d}/{:3<d}'.format(step + 1, samples_num),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

    """Write to file"""
    if args.mid_out_path:
        out = np.reshape(waveform, (-1, 3))[:, :2]
        op = args.wav_seed if init_chain else args.mid_out_path
        convert_to_mid(out, sample_rate, filename=args.mid_out_path, op=op, chain_mel=chain_mel, chain_vel=chain_vel)

    if chain_mel and not init_chain:
        kl_result.append(batch_entropy(pre_pred, cur_pred))
        pre_pred = cur_pred
    sess.close()
    print('Finished generating.')

if __name__ == '__main__':
    idx = 0
    pre_pred = []
    kl_result = []
    variables_to_restore = {}
    args = get_arguments()

    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)
    for idx in range(args.num_iterations):
        chain_mel = chain_vel = init_chain = False
        if idx%2==0:
            chain_mel = True
            ckpt = args.checkpoint_mel
            if idx==0:
                init_chain=True
        else:
            chain_vel = True
            ckpt = args.checkpoint_vel
        generate_func(args, wavenet_params, chain_mel, chain_vel, init_chain)
    print(kl_result)

