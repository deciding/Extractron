import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper

global_seed=None

class ExtractTestHelper(Helper):
    def __init__(self, batch_size, stop_length, mixed_spec, spkid_embedding, hparams):
        with tf.name_scope('ExtractTestHelper'):
            self._batch_size = batch_size
            self._output_dim = hparams.num_mels
            self._length = stop_length
            self._mixed_spec = mixed_spec
            self._spkid_embedding = spkid_embedding

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        initial_input=_go_frames(self._batch_size, self._output_dim)
        initial_input=tf.concat(
                [initial_input, self._mixed_spec[:, 0, :], self._spkid_embedding], axis=-1)
        return (tf.tile([False], [self._batch_size]), initial_input)

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        '''Stop on EOS. Otherwise, pass the last output as the next input and pass through state.'''
        with tf.name_scope('ExtractTestHelper'):
            finished = (time + 1 >= self._length)
            next_inputs = outputs[:, -self._output_dim:]
            next_inputs = tf.cond(
                    tf.less(time, self._length-1),
                    lambda: tf.concat(
                    [next_inputs, self._mixed_spec[:, time+1, :], self._spkid_embedding],
                    axis=-1),
                    lambda: tf.concat(
                    [next_inputs, self._mixed_spec[:, self._length-1, :], self._spkid_embedding],
                    axis=-1))
            next_state = state
            return (finished, next_inputs, next_state)

class ExtractTrainingHelper(Helper):
    def __init__(self, batch_size, mixed_spec, target_spec, spkid_embedding, hparams, global_step):
        # inputs is [N, T_in], targets is [N, T_out, D]
        with tf.name_scope('ExtractTrainingHelper'):
            self._batch_size = batch_size
            self._output_dim = hparams.num_mels
            self._ratio = tf.convert_to_tensor(
                hparams.extractron_teacher_forcing_ratio)
            self._hparams = hparams
            self.global_step = global_step
            self._mixed_spec = mixed_spec
            self._target_spec = target_spec
            self._spkid_embedding = spkid_embedding
            self._length = tf.shape(target_spec)[1]

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        if self._hparams.extractron_teacher_forcing_mode == 'scheduled':
            self._ratio = _teacher_forcing_ratio_decay(
                    self._hparams.extractron_teacher_forcing_init_ratio,
                    self.global_step, self._hparams)

        initial_input=_go_frames(self._batch_size, self._output_dim)
        initial_input=tf.concat(
                [initial_input, self._mixed_spec[:, 0, :], self._spkid_embedding], axis=-1)
        return (tf.tile([False], [self._batch_size]), initial_input)

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name or 'ExtractTrainingHelper'):
            # synthesis stop (we let the model see paddings as we mask them when computing loss functions)
            finished = (time + 1 >= self._length)

            # Pick previous outputs randomly with respect to teacher forcing ratio
            next_inputs = tf.cond(
                tf.less(tf.random_uniform([], minval=0, maxval=1,
                                          dtype=tf.float32, seed=global_seed), self._ratio),
                # Teacher-forcing: return true frame
                lambda: self._target_spec[:, time, :],
                lambda: outputs[:, -self._output_dim:])

            next_inputs = tf.cond(
                    tf.less(time, self._length-1),
                    lambda: tf.concat(
                    [next_inputs, self._mixed_spec[:, time+1, :], self._spkid_embedding],
                    axis=-1),
                    lambda: tf.concat(
                    [next_inputs, self._mixed_spec[:, self._length-1, :], self._spkid_embedding],
                    axis=-1))

            # Pass on state
            next_state = state
            return (finished, next_inputs, next_state)


def _go_frames(batch_size, output_dim):
    '''Returns all-zero <GO> frames for a given batch size and output dimension'''
    return tf.tile([[0.0]], [batch_size, output_dim])


def _teacher_forcing_ratio_decay(init_tfr, global_step, hparams):
        #################################################################
        # Narrow Cosine Decay:

        # Phase 1: tfr = 1
        # We only start learning rate decay after 10k steps

        # Phase 2: tfr in ]0, 1[
        # decay reach minimal value at step ~280k

        # Phase 3: tfr = 0
        # clip by minimal teacher forcing ratio value (step >~ 280k)
        #################################################################
        # Compute natural cosine decay
    tfr = tf.train.cosine_decay(init_tfr,
                                global_step=global_step -
                                hparams.extractron_teacher_forcing_start_decay,  # tfr = 1 at step 10k
                                decay_steps=hparams.extractron_teacher_forcing_decay_steps,  # tfr = 0 at step ~280k
                                # tfr = 0% of init_tfr as final value
                                alpha=hparams.extractron_teacher_forcing_decay_alpha,
                                name='tfr_cosine_decay')

    # force teacher forcing ratio to take initial value when global step < start decay step.
    narrow_tfr = tf.cond(
        tf.less(global_step, tf.convert_to_tensor(
            hparams.extractron_teacher_forcing_start_decay)),
        lambda: tf.convert_to_tensor(init_tfr),
        lambda: tfr)

    return narrow_tfr
