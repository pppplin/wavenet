"""
Utility function for converting an audio file
to a pretty_midi.PrettyMIDI object. Note that this method is nowhere close
to the state-of-the-art in automatic music transcription.
This just serves as a fun example for rough
transcription which can be expanded on for anyone motivated.
"""
from __future__ import division
import numpy as np
import pretty_midi
import copy as cp
import tensorflow as tf
import librosa
from wavenet import mu_law_encode, audio_reader
from wavenet.audio_reader import load_audio_velocity

def array_to_pretty_midi(array, fs=100, velocity=90, program=68, op=None):
    """
    Convert an array (-1, ) numpy array to a PrettyMidi object
    if velocity not included: (-1, ), else (-1, 2)
    """
    n = array.shape[0]
    #fix the tempo bug
    if not op:
        pm = pretty_midi.PrettyMIDI()
    else:
        opm = pretty_midi.PrettyMIDI(op)
        pm = cp.deepcopy(opm)

    instrument = pretty_midi.Instrument(program=program)
    ts = 0
    te = 1/fs
    p = array[0]
    #print(array, p)
    for i in range(n-1):
        if array[i+1]!=array[i] and p!=0:
            pm_note = pretty_midi.Note(
                velocity=velocity,
                pitch=int(p),
                start=ts,
                end=te)
            instrument.notes.append(pm_note)
            ts = te
            te = ts + 1/fs
            p = array[i+1]
        elif p==0 and array[i+1]!=array[i]:
            ts = te
            te = ts + 1/fs
            p = array[i+1]
        else:
            te += 1/fs

    pm.instruments.append(instrument)
    return pm

def array_to_pretty_midi_velocity(array, fs=100, program=0,
        op=None, vel_set=None, forget=False):
    """
    Convert an array (-1, 2) numpy array to a PrettyMidi object
    """
    # take avg of all velocities not 0
    n = array.shape[0]
    if not op:
        pm = pretty_midi.PrettyMIDI()
    else:
        opm = pretty_midi.PrettyMIDI(op)
        pm = cp.deepcopy(opm)
    m = array[:, 0]
    v = array[:, 1]
    #nan sometimes
    v = np.nan_to_num(v)
    instrument = pretty_midi.Instrument(program=program)
    ts = 0
    te = 1/fs
    p = m[0]
    if vel_set is None:
        cur_v_sum = v[0]
        cur_v_valid = 1 if v[0]!=0 else 0
        for i in range(n-1):
            if m[i+1]!=m[i]:
                if p!=0:
                    if cur_v_valid==0:
                        #set default 100
                        v_avg = 100
                    else:
                        v_avg = int(np.nan_to_num(np.rint(cur_v_sum/cur_v_valid)))
                    pm_note = pretty_midi.Note(
                        velocity=v_avg,
                        pitch=int(p),
                        start=ts,
                        end=te)
                    instrument.notes.append(pm_note)
                ts = te
                te = ts + 1/fs
                p = m[i+1]
                cur_v_valid = 1 if v[i+1]!=0 else 0
                cur_v_sum = v[i+1]
            else:
                te += 1/fs
                if p!=0:
                    cur_v_sum += v[i+1]
                    if v[i+1]!=0:
                        cur_v_valid += 1
        pm.instruments.append(instrument)
        print("Forget NOT INPLEMENTED!")
    else:
        cur_v_set = []
        for i in range(n-1):
            if m[i+1]!=m[i]:
                if p!=0:
                    if cur_v_set==[]:
                        v_major = 95
                    else:
                        v_major = max(set(cur_v_set), key=cur_v_set.count)
                    pm_note = pretty_midi.Note(
                        velocity=v_major,
                        pitch=int(p),
                        start=ts,
                        end=te)
                    instrument.notes.append(pm_note)
                ts = te
                te = ts + 1/fs
                p = m[i+1]
                if int(v[i+1])!=0:
                    cur_v_set = [int(v[i+1])]
            else:
                te += 1/fs
                if p!=0 and int(v[i+1])!=0:
                    cur_v_set.append(int(v[i+1]))
        if not forget:
            pm.instruments.append(instrument)
        else:
            pm.instruments = pm.instruments[:3]
            pm.instruments.append(instrument)
    return pm

def create_midi_seed(filename, samples_num, sample_rate, window_size,
        vel_set=None, silence_threshold=0, local_sample_rate=None, use_velocity=False,
        use_chord=False, chain_mel=False, chain_vel=False, init=False):
    if vel_set is not None:
        default_velocity=90
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
    if init:
        last=False
    else:
        last=True
    velocity = load_audio_velocity(filename, sample_rate, last=last)
    velocity = np.reshape(velocity, (-1, 1))
    if vel_set is not None:
        velocity[velocity==0] = default_velocity
    pm.instruments = chord_instrument
    chords = pm.get_piano_roll(fs=sample_rate, times=None)
    chords = np.swapaxes(chords, 0, 1)
    chords = np.argmax(chords, axis=-1)
    chords = np.reshape(chords, (-1, 1))
    if chain_mel or chain_vel:
        samples_num = np.size(chords) - np.size(midi)
    midi_size = np.size(midi)
    if init:
        #80 95 105
        #originally 95, now random
        #vel_init = np.random.randint(4, size=(np.size(chords)-midi_size, 1))
        #vel_init[vel_init==1] = 80
        #vel_init[vel_init==2] = 95
        #vel_init[vel_init==3] = 105
        vel_init = np.asarray([95]*(np.size(chords)-midi_size))
        vel_init = np.reshape(vel_init, (-1, 1))
        velocity = np.concatenate((velocity, vel_init), axis=0)
    #chords = chords[:min_len]

    cut_index = np.size(midi) if np.size(midi)<window_size else window_size
    if use_velocity:
        prod = np.concatenate((midi, velocity), axis=1)
        return tf.stack(prod[:cut_index])
    if use_chord:
        chords_1 = chords[cut_index: cut_index+samples_num]
        prod = np.concatenate((midi[:cut_index], chords[:cut_index]), axis=1)
        return tf.stack(prod), chords_1
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
    if local_sample_rate:
        #TODO stack!!
        raise ValueError("stack not working for local condition")
        cmidi = pretty_midi.PrettyMIDI(filename)
        cmidi.instruments = [cmidi.instruments[-1]]
        chords = cmidi.get_piano_roll(fs = local_sample_rate, times = None)
        chords = np.swapaxes(chords, 0, 1)
        chords = np.argmax(chords, axis=-1)
        rate = sample_rate/local_sample_rate
        chords_0 = chords[:midi.shape[0]/rate]
        chords_1 = chords[cut_index/rate: cut_index/rate+samples_num]
        midi = np.reshape(midi, (-1, 1))
        chords_0 = np.reshape(chords_0, (-1, 1))
        chords_1 = np.reshape(chords_1, (-1, 1))
        midi = np.concatenate((midi, chords_0), axis=1)
        return tf.stack(midi[:cut_index]), chords_1

    return tf.stack(midi[:cut_index])

def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))

def convert_to_mid(array, sample_rate, filename, op, vel_set=None,
        velocity=False, chain_mel=False, chain_vel=False, forget=False, program=None):
    """
    array: np.ndarray, shape = (?, 1), directed generated array
    save as midi file
    """
    arr = np.asarray(array)
    if chain_vel:
        arr = arr[:, [1, 0]]
    if velocity or chain_mel or chain_vel:
        if chain_vel and program is None:
            program = 0
        elif (chain_mel or velocity) and program is None:
            program = 68
        mid = array_to_pretty_midi_velocity(arr, fs=sample_rate, program=program,
                op=op, vel_set=vel_set, forget=forget)
    else:
        mid = array_to_pretty_midi(arr, fs=sample_rate, program=0, op=op)
    mid.write(filename)

def create_seed(filename,
                sample_rate,
                quantization_channels,
                window_size,
                silence_threshold=0):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio_reader.trim_silence(audio, silence_threshold)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))

    return quantized[:cut_index]


