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

def array_to_pretty_midi(array, fs = 100, velocity = 90, program = 68, op = None):
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

def array_to_pretty_midi_velocity(array, fs = 100, program = 68, op = None):
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
    return pm

