import numpy as np
from tensorflow.contrib.seq2seq import dynamic_decode
import tensorflow as tf

from infolog import log
from extractron.models.Architecture_wrappers import ExtractronDecoderCell
from extractron.models.custom_decoder import CustomDecoder
from extractron.models.helpers import ExtractTrainingHelper, ExtractTestHelper
from extractron.models.modules import Prenet, DecoderRNN, FrameProjection, Postnet, CBHG
global_seed=None

class Extractron():
    """Extractron-2 Feature prediction Model.
    """

    def __init__(self, hparams, args=None):
        self._hparams = hparams
        self._args = args

    def initialize(self, mixed_mel=None, target_mel=None, mixed_phase=None, target_phase=None, mixed_linear=None, target_linear=None, spkid_embeddings=None,
                   global_step=None, is_training=False):

        hp = self._hparams

        self.tower_mixed_mel = []
        self.tower_target_mel = []
        self.tower_mixed_phase = []
        self.tower_target_phase = []
        self.tower_mixed_linear = []
        self.tower_target_linear = []
        self.tower_spkid_embeddings = []

        batch_size = tf.shape(mixed_mel)[0] // hp.extractron_num_gpus
        #spec_channels = hp.num_freq
        spec_channels = hp.num_mels
        spkid_embedding_dim = hp.spkid_embedding_dim

        #split to tower
        #fill in to the tower
        for i in range(hp.extractron_num_gpus):
            self.tower_mixed_mel.append(
                    tf.reshape(
                        mixed_mel[i*batch_size:(i+1)*batch_size],
                        [batch_size, -1, spec_channels]))
            self.tower_target_mel.append(
                    tf.reshape(
                        target_mel[i*batch_size:(i+1)*batch_size],
                        [batch_size, -1, spec_channels]))
            cur_spkid_embeddings = tf.reshape(
                    spkid_embeddings[i*batch_size:(i+1)*batch_size],
                    [batch_size, spkid_embedding_dim])
            if mixed_phase is not None:
                self.tower_mixed_phase.append(
                        tf.reshape(
                            mixed_phase[i*batch_size:(i+1)*batch_size],
                            [batch_size, -1, hp.num_freq]))
            if target_phase is not None:
                self.tower_target_phase.append(
                        tf.reshape(
                            target_phase[i*batch_size:(i+1)*batch_size],
                            [batch_size, -1, hp.num_freq]))
            if mixed_linear is not None:
                self.tower_mixed_linear.append(
                        tf.reshape(
                            mixed_linear[i*batch_size:(i+1)*batch_size],
                            [batch_size, -1, hp.num_freq]))
            if target_linear is not None:
                self.tower_target_linear.append(
                        tf.reshape(
                            target_linear[i*batch_size:(i+1)*batch_size],
                            [batch_size, -1, hp.num_freq]))

            #cur_spkid_embeddings = tf.expand_dims(
            #    cur_spkid_embeddings, 1)
            ##batch * L * 256
            #cur_spkid_embeddings = tf.tile(
            #    cur_spkid_embeddings, [1, tf.shape(mixed_spec)[1], 1])
            self.tower_spkid_embeddings.append(cur_spkid_embeddings)

        #TODO:save intermidiate tower result
        self.tower_decoder_outputs=[]
        self.tower_residual=[]
        self.tower_projected_residual=[]
        self.tower_predict_outputs=[]
        self.tower_linear_outputs=[]

        gpus = ["/gpu:{}".format(i) for i in range(hp.extractron_gpu_start_idx,
                                                   hp.extractron_gpu_start_idx+hp.extractron_num_gpus)]
        for i in range(hp.extractron_num_gpus):
            with tf.device(
                    tf.train.replica_device_setter(
                        ps_tasks=1, ps_device="/cpu:0", worker_device=gpus[i])):
                with tf.variable_scope('inference'):

                    #TODO: hparams
                    prenet = Prenet(is_training, layers_sizes=hp.prenet_layers,
                                    drop_rate=hp.extractron_dropout_rate, scope='decoder_prenet')

                    # Frames Projection layer
                    #TODO: one frame a time
                    frame_projection = FrameProjection(
                        spec_channels, scope='linear_transform_projection')

                    decoder_lstm = DecoderRNN(is_training,
                            layers=hp.decoder_layers,
                            size=hp.decoder_lstm_units,
                            zoneout=hp.extractron_zoneout_rate,
                            scope='decoder_LSTM')

                    # Decoder Cell ==> [batch_size, decoder_steps, num_mels * r] (after decoding)
                    #TODO: delete attention
                    #TODO: convolve the context of noisy speech
                    decoder_cell = ExtractronDecoderCell(
                        prenet,
                        decoder_lstm,
                        frame_projection)

                    if hp.bidirection:
                        reversed_decoder_lstm = DecoderRNN(is_training,
                                layers=hp.decoder_layers,
                                size=hp.decoder_lstm_units,
                                zoneout=hp.extractron_zoneout_rate,
                                scope='reversed_decoder_LSTM')

                        # Decoder Cell ==> [batch_size, decoder_steps, num_mels * r] (after decoding)
                        #TODO: delete attention
                        #TODO: convolve the context of noisy speech
                        reversed_decoder_cell = ExtractronDecoderCell(
                            prenet,
                            reversed_decoder_lstm,
                            frame_projection)


                    # Define the helper for our decoder
                    #TODO: since we are using generative method, thus we need the last output as input
                    if is_training:
                        self.helper = ExtractTrainingHelper(
                            batch_size,
                            self.tower_mixed_mel[i],
                            self.tower_target_mel[i],
                            self.tower_spkid_embeddings[i],
                            hp, global_step)
                    else:
                        self.helper = ExtractTestHelper(
                            batch_size,
                            tf.shape(self.tower_mixed_mel[i])[1],
                            self.tower_mixed_mel[i],
                            self.tower_spkid_embeddings[i],
                            hp)

                    if hp.bidirection:
                        reversed_mixed_mel = tf.reverse(self.tower_mixed_mel[i], [1])
                        reversed_target_mel = tf.reverse(self.tower_target_mel[i], [1])
                        if is_training:
                            self.reversed_helper = ExtractTrainingHelper(
                                batch_size,
                                reversed_mixed_mel,
                                reversed_target_mel,
                                self.tower_spkid_embeddings[i],
                                hp, global_step)
                        else:
                            self.reversed_helper = ExtractTestHelper(
                                batch_size,
                                tf.shape(self.tower_mixed_mel[i])[1],
                                reversed_mixed_mel,
                                self.tower_spkid_embeddings[i],
                                hp)

                    # initial decoder state
                    decoder_init_state = decoder_cell.zero_state(
                        batch_size=batch_size, dtype=tf.float32)

                    # Only use max iterations at synthesis time
                    max_iters = hp.max_iters if not is_training else None

                    # Decode
                    #TODO: change to lstm instead of dynamic decoder
                    (frames_prediction, _), final_decoder_state, _ = dynamic_decode(
                        CustomDecoder(decoder_cell, self.helper,
                                      decoder_init_state),
                        impute_finished=hp.impute_finished,
                        maximum_iterations=max_iters,
                        swap_memory=hp.extractron_swap_with_cpu)

                    decoder_output = tf.reshape(
                        frames_prediction, [batch_size, -1, spec_channels])

                    if hp.bidirection:
                        (reversed_frames_prediction, _), reversed_final_decoder_state, _ = dynamic_decode(
                            CustomDecoder(reversed_decoder_cell, self.reversed_helper,
                                          decoder_init_state),
                            impute_finished=hp.impute_finished,
                            maximum_iterations=max_iters,
                            swap_memory=hp.extractron_swap_with_cpu)

                        reversed_decoder_output = tf.reshape(
                            reversed_frames_prediction, [batch_size, -1, spec_channels])
                        decoder_output = tf.concat([
                            decoder_output,
                            tf.reverse(reversed_decoder_output, [1])
                            ],
                            axis=-1)
                        bidirection_projection = FrameProjection(
                            spec_channels, scope='bidirection_projection')
                        decoder_output = bidirection_projection(decoder_output)

                    # Postnet
                    postnet = Postnet(is_training, hparams=hp,
                                      scope='postnet_convolutions')

                    # Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
                    #TODO: postnet maybe necessary for mask has linear attrinute
                    residual = postnet(decoder_output)

                    # Project residual to same dimension as mel spectrogram
                    # ==> [batch_size, decoder_steps * r, num_mels]
                    residual_projection = FrameProjection(
                        spec_channels, scope='postnet_projection')
                    projected_residual = residual_projection(residual)

                    # Compute the mel spectrogram
                    predict_outputs = decoder_output + projected_residual

                    post_cbhg = CBHG(hp.cbhg_kernels, hp.cbhg_conv_channels, hp.cbhg_pool_size, [hp.cbhg_projection, spec_channels],
                                     hp.cbhg_projection_kernel_size, hp.cbhg_highwaynet_layers,
                                     hp.cbhg_highway_units, hp.cbhg_rnn_units, is_training, name='CBHG_postnet')

                    #[batch_size, decoder_steps(mel_frames), cbhg_channels]
                    post_outputs = post_cbhg(predict_outputs, None)

                    # Linear projection of extracted features to make linear spectrogram
                    linear_specs_projection = FrameProjection(
                        hp.num_freq, scope='cbhg_linear_specs_projection')

                    linear_outputs = linear_specs_projection(post_outputs)

                    #saving intermidiate results
                    self.tower_decoder_outputs.append(decoder_output)
                    self.tower_residual.append(residual)
                    self.tower_projected_residual.append(projected_residual)
                    self.tower_predict_outputs.append(predict_outputs)
                    self.tower_linear_outputs.append(linear_outputs)
            log('initialisation done {}'.format(gpus[i]))
        #============END of all gpus initialization=====================

        if is_training:
            self.ratio = self.helper._ratio
        self.all_vars = tf.trainable_variables()

        log('Initialized Extractron model. Dimensions (? = dynamic shape): ')
        log('  Train mode:               {}'.format(is_training))
        for i in range(hp.extractron_num_gpus):
            log('  device:                   {}'.format(
                i+hp.extractron_gpu_start_idx))
            log('  decoder out:              {}'.format(
                self.tower_decoder_outputs[i].shape))
            log('  residual out:             {}'.format(
                self.tower_residual[i].shape))
            log('  projected residual out:   {}'.format(
                self.tower_projected_residual[i].shape))
            log('  predicted out:                  {}'.format(
                self.tower_predict_outputs[i].shape))
            log('  linear out:                  {}'.format(
                self.tower_linear_outputs[i].shape))

            # 1_000_000 is causing syntax problems for some people?! Python please :)
            log('  Extractron Parameters       {:.3f} Million.'.format(
                np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))

    def l1_loss(self, target, predict):
        l1 = tf.abs(predict - target)
        linear_loss = tf.reduce_mean(l1)
        return linear_loss

    def add_loss(self, global_step=None):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        hp = self._hparams

        self.tower_before_loss = []
        self.tower_after_loss = []
        self.tower_linear_loss = []
        self.tower_regularization_loss = []
        self.tower_loss = []

        total_before_loss = 0
        total_after_loss = 0
        total_linear_loss = 0
        total_regularization_loss = 0
        total_loss = 0

        gpus = ["/gpu:{}".format(i) for i in range(hp.extractron_gpu_start_idx,
                                                   hp.extractron_gpu_start_idx+hp.extractron_num_gpus)]

        for i in range(hp.extractron_num_gpus):
            with tf.device(
                    tf.train.replica_device_setter(
                        ps_tasks=1, ps_device="/cpu:0", worker_device=gpus[i])):
                with tf.variable_scope('loss'):
                    # Compute loss of predictions before postnet
                    before = tf.losses.mean_squared_error(
                        self.tower_target_mel[i], self.tower_decoder_outputs[i])
                    after = tf.losses.mean_squared_error(
                        self.tower_target_mel[i], self.tower_predict_outputs[i])
                    linear_loss = tf.losses.mean_squared_error(
                        self.tower_target_linear[i], self.tower_linear_outputs[i])
                    # Compute the regularization weight
                    reg_weight = hp.extractron_reg_weight

                    # Regularize variables
                    # Exclude all types of bias, RNN (Bengio et al. On the difficulty of training recurrent neural networks), embeddings and prediction projection layers.
                    # Note that we consider attention mechanism v_a weights as a prediction projection layer and we don't regularize it. (This gave better stability)
                    regularization = tf.add_n([tf.nn.l2_loss(v) for v in self.all_vars
                                               if not('bias' in v.name
                                                   or 'Bias' in v.name
                                                   or '_projection' in v.name
                                                   or 'RNN' in v.name
                                                   or 'LSTM' in v.name
                                                      )]) * reg_weight


                    # Compute final loss term
                    self.tower_before_loss.append(before)
                    self.tower_after_loss.append(after)
                    self.tower_linear_loss.append(linear_loss)
                    self.tower_regularization_loss.append(regularization)

                    loss = before + after + linear_loss + regularization
                    self.tower_loss.append(loss)

        for i in range(hp.extractron_num_gpus):
            total_before_loss += self.tower_before_loss[i]
            total_after_loss += self.tower_after_loss[i]
            total_linear_loss += self.tower_linear_loss[i]
            total_regularization_loss += self.tower_regularization_loss[i]
            total_loss += self.tower_loss[i]

        self.before_loss = total_before_loss / hp.extractron_num_gpus
        self.after_loss = total_after_loss / hp.extractron_num_gpus
        self.linear_loss = total_linear_loss / hp.extractron_num_gpus
        self.regularization_loss = total_regularization_loss / hp.extractron_num_gpus
        self.loss = total_loss / hp.extractron_num_gpus

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
        Args:
                global_step: int32 scalar Tensor representing current global step in training
        '''
        hp = self._hparams
        tower_gradients = []

        # 1. Declare GPU Devices
        gpus = ["/gpu:{}".format(i) for i in range(hp.extractron_gpu_start_idx,
                                                   hp.extractron_gpu_start_idx + hp.extractron_num_gpus)]

        grad_device = '/cpu:0' if hp.extractron_num_gpus > 1 else gpus[0]

        with tf.device(grad_device):
            with tf.variable_scope('optimizer'):
                if hp.extractron_decay_learning_rate:
                    self.decay_steps = hp.extractron_decay_steps
                    self.decay_rate = hp.extractron_decay_rate
                    self.learning_rate = self._learning_rate_decay(
                        hp.extractron_initial_learning_rate, global_step)
                else:
                    self.learning_rate = tf.convert_to_tensor(
                        hp.extractron_initial_learning_rate)

                optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.extractron_adam_beta1,
                                                   hp.extractron_adam_beta2, hp.extractron_adam_epsilon)

        trainable_vars = tf.trainable_variables(scope=None)

        # 2. Compute Gradient
        for i in range(hp.extractron_num_gpus):
            #  Device placement
            with tf.device(
                    tf.train.replica_device_setter(
                        ps_tasks=1, ps_device="/cpu:0", worker_device=gpus[i])):
                with tf.variable_scope('optimizer'):
                    gradients = optimizer.compute_gradients(self.tower_loss[i], trainable_vars)
                    tower_gradients.append(gradients)

        # 3. Average Gradient
        with tf.device(grad_device):
            avg_grads = []
            vars = []
            for grad_and_vars in zip(*tower_gradients):
                # grads_vars = [(grad1, var), (grad2, var), ...]

                if grad_and_vars[0][0] is None:
                    print("None grad(during training of mel autoencoder, mainly from text encoder): ", grad_and_vars[0][1])
                    continue
                grads = []
                for g, v in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)
                    # Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)

                v = grad_and_vars[0][1]
                avg_grads.append(grad)
                vars.append(v)

            self.gradients = avg_grads
            # Just for causion
            # https://github.com/Rayhane-mamah/Tacotron-2/issues/11
            if hp.extractron_clip_gradients:
                clipped_gradients, _ = tf.clip_by_global_norm(
                    avg_grads, 1.)  # __mark 0.5 refer
            else:
                clipped_gradients = avg_grads

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, vars),
                                                          global_step=global_step)


    def _learning_rate_decay(self, init_lr, global_step):
        #################################################################
        # Narrow Exponential Decay:

        # Phase 1: lr = 1e-3
        # We only start learning rate decay after 50k steps

        # Phase 2: lr in ]1e-5, 1e-3[
        # decay reach minimal value at step 310k

        # Phase 3: lr = 1e-5
        # clip by minimal learning rate value (step > 310k)
        #################################################################
        hp = self._hparams

        # Compute natural exponential decay
        lr = tf.train.exponential_decay(init_lr,
                                        global_step - hp.extractron_start_decay,  # lr = 1e-3 at step 50k
                                        self.decay_steps,
                                        self.decay_rate,  # lr = 1e-5 around step 310k
                                        name='lr_exponential_decay')

        # clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, hp.extractron_final_learning_rate), init_lr)
