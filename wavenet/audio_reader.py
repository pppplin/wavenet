import fnmatch
import os
import random
import re
import threading

import librosa
import numpy as np
import tensorflow as tf
import pretty_midi

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.mid'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate, lc_enabled=False, load_velocity=False, load_chord=False, chain_mel=False, chain_vel=False, local_sample_rate=None):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])

        midi = pretty_midi.PrettyMIDI(filename)
        melody_instrument = midi.instruments[0]

        if load_chord or lc_enabled or chain_mel or chain_vel:
            chord_instrument = midi.instruments[-1]
            midi.instruments = [chord_instrument]
            if lc_enabled:
                chord = midi.get_piano_roll(fs = local_sample_rate, times = None)
            else:
                chord = midi.get_piano_roll(fs = sample_rate, times = None)
            chord = np.swapaxes(chord, 0, 1)
            chord = np.argmax(chord, axis=-1)
            chord = np.reshape(chord, (-1, 1))

        midi.instruments = [melody_instrument]
        midi = midi.get_piano_roll(fs = sample_rate, times = None)
        midi = np.swapaxes(midi, 0, 1)
        midi = np.argmax(midi, axis = -1)
        midi = np.reshape(midi, (-1, 1))
        velocity = load_audio_velocity(filename, sample_rate)
        velocity = np.reshape(velocity, (-1, 1))
        if not lc_enabled:
            cut_length = min(midi.shape[0], velocity.shape[0], chord.shape[0])
            midi = midi[:cut_length, :]
            velocity = velocity[:cut_length, :]
            chord = chord[:cut_length, :]
        else:
            rate = sample_rate/local_sample_rate
            cut_length = min(midi.shape[0], category_id.shape[0]*rate)
            midi = midi[:cut_length, :]
            cut_length = cut_length/rate
            chord = chord[:cut_length, :]

        if load_velocity:
            prod = np.concatenate((midi, velocity), axis=1)
            yield prod, filename, category_id
        elif chain_mel:
            cond = np.concatenate((velocity, chord), axis=1)
            yield midi, filename, cond
        elif chain_vel:
            cond = np.concatenate((midi, chord), axis=1)
            yield velocity, filename, cond
        elif load_velocity and load_chord:
            raise ValueError("Does not support velocity and chord")
        elif load_chord:
            yield midi, filename, chord
        else:
            yield midi, filename, category_id

def load_audio_velocity(filename, sample_rate, chain_vel=False):
    """
    filename: must be midi
    return: nd array (, 1), same shape as piano_roll, serves as input
    """
    pm = pretty_midi.PrettyMIDI(filename)
    if chain_vel:
        melody_instrument = pm.instruments[-1]
    else:
        melody_instrument = pm.instruments[0]
    pm.instruments = [melody_instrument]
    T = pm.get_piano_roll(fs = sample_rate).shape[1]
    velocity = np.zeros((T, 1))
    notes = pm.instruments[0].notes
    fs = 1.0/sample_rate
    for n in notes:
        start_idx = int(n.start/fs)
        end_idx = int(n.end/fs)
        velocity[start_idx: end_idx, 0] = n.velocity
    return velocity

def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 local_sample_rate,
                 gc_enabled,
                 lc_enabled,
                 load_velocity,
                 load_chord,
                 chain_mel,
                 chain_vel,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.lc_enabled = lc_enabled
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.load_velocity = load_velocity
        self.load_chord = load_chord
        self.chain_mel = chain_mel
        self.chain_vel = chain_vel
        self.local_sample_rate = local_sample_rate
        if self.load_velocity:
            self.queue = tf.PaddingFIFOQueue(queue_size, ['float32'], shapes=[(None, 2)])
        else:
            self.queue = tf.PaddingFIFOQueue(queue_size, ['float32'], shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        if self.chain_mel or self.chain_vel:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'], shapes=[(None, 2)])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])
        elif self.load_chord:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'], shapes=[(None, 1)])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])
        elif self.gc_enabled and not self.load_chord:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
        if self.gc_enabled and not_all_have_id(files) and not self.load_chord and not self.chain_mel and not self.chain_vel:
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled and not self.load_chord:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_audio(self.audio_dir, self.sample_rate, self.lc_enabled, self.load_velocity,
                    self.load_chord, self.chain_mel, self.chain_vel, self.local_sample_rate)
            for audio, filename, category_id in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                #TODO
                if self.silence_threshold is not None and self.silence_threshold!=0:
                    # Remove silence
                    audio = trim_silence(audio[:, 0], self.silence_threshold)
                    audio = audio.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))
                #velocity included and padded
                audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]],
                               'constant')
                #chord padding
                if self.load_chord or self.chain_mel or self.chain_vel:
                    category_id = np.pad(category_id, [[self.receptive_field, 0], [0, 0]], 'constant')

                if self.sample_size:
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    while len(audio) > self.receptive_field:
                        piece = audio[:(self.receptive_field +
                                        self.sample_size), :]
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        audio = audio[self.sample_size:, :]
                        if self.chain_mel or self.chain_mel or self.load_chord:
                            category_id_piece = category_id[:(self.receptive_field+self.sample_size), :]
                            sess.run(self.gc_enqueue, feed_dict={
                                self.id_placeholder: category_id_piece})
                            category_id = category_id[self.sample_size:, :]
                        elif self.gc_enabled:
                            sess.run(self.gc_enqueue, feed_dict={self.id_placeholder: category_id})

                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: audio})
                    if self.gc_enabled:
                        sess.run(self.gc_enqueue,
                                 feed_dict={self.id_placeholder: category_id})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
