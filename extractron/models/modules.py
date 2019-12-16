import tensorflow as tf

global_seed=None
use_cudnn=False

class HighwayNet:
    def __init__(self, units, name=None):
        self.units = units
        self.scope = 'HighwayNet' if name is None else name

        self.H_layer = tf.layers.Dense(
            units=self.units, activation=tf.nn.relu, name='H')
        self.T_layer = tf.layers.Dense(
            units=self.units, activation=tf.nn.sigmoid, name='T', bias_initializer=tf.constant_initializer(-1.))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            H = self.H_layer(inputs)
            T = self.T_layer(inputs)
            return H * T + inputs * (1. - T)


class CBHG:
    def __init__(self, K, conv_channels, pool_size, projections, projection_kernel_size, n_highwaynet_layers, highway_units, rnn_units, is_training, name=None):
        self.K = K
        self.conv_channels = conv_channels
        self.pool_size = pool_size

        self.projections = projections
        self.projection_kernel_size = projection_kernel_size

        self.is_training = is_training
        self.scope = 'CBHG' if name is None else name

        self.highway_units = highway_units
        self.highwaynet_layers = [HighwayNet(highway_units, name='{}_highwaynet_{}'.format(
            self.scope, i+1)) for i in range(n_highwaynet_layers)]
        if not use_cudnn:
            self._fw_cell = tf.nn.rnn_cell.GRUCell(
                rnn_units, name='{}_forward_RNN'.format(self.scope))
            self._bw_cell = tf.nn.rnn_cell.GRUCell(
                rnn_units, name='{}_backward_RNN'.format(self.scope))
        else:
            self._fw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(rnn_units)
            self._bw_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(rnn_units)

    def __call__(self, inputs, input_lengths):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('conv_bank'):
                # Convolution bank: concatenate on the last axis to stack channels from all convolutions
                # The convolution bank uses multiple different kernel sizes to have many insights of the input sequence
                # This makes one of the strengths of the CBHG block on sequences.
                conv_outputs = tf.concat(
                    [conv1d(inputs, k, self.conv_channels, tf.nn.relu, self.is_training,
                            0., 'conv1d_{}'.format(k)) for k in range(1, self.K+1)],
                    axis=-1
                )

            # Maxpooling (dimension reduction, Using max instead of average helps finding "Edges" in mels)
            maxpool_output = tf.layers.max_pooling1d(
                conv_outputs,
                pool_size=self.pool_size,
                strides=1,
                padding='same')

            # Two projection layers
            proj1_output = conv1d(maxpool_output, self.projection_kernel_size,
                                  self.projections[0], tf.nn.relu, self.is_training, 0., 'proj1')
            proj2_output = conv1d(proj1_output, self.projection_kernel_size,
                                  self.projections[1], lambda _: _, self.is_training, 0., 'proj2')

            # Residual connection
            highway_input = proj2_output + inputs

            # Additional projection in case of dimension mismatch (for HighwayNet "residual" connection)
            if highway_input.shape[2] != self.highway_units:
                highway_input = tf.layers.dense(
                    highway_input, self.highway_units)

            # 4-layer HighwayNet
            for highwaynet in self.highwaynet_layers:
                highway_input = highwaynet(highway_input)
            rnn_input = highway_input

            # Bidirectional RNN
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                self._fw_cell,
                self._bw_cell,
                rnn_input,
                sequence_length=input_lengths,
                dtype=tf.float32)
            # Concat forward and backward outputs
            return tf.concat(outputs, axis=2)

class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
    '''Wrapper for tf LSTM to create Zoneout LSTM Cell

    inspired by:
    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py

    Published by one of 'https://arxiv.org/pdf/1606.01305.pdf' paper writers.

    Many thanks to @Ondal90 for pointing this out. You sir are a hero!
    '''

    def __init__(self, num_units, is_training, zoneout_factor_cell=0., zoneout_factor_output=0., state_is_tuple=True, name=None):
        '''Initializer with possibility to set different zoneout values for cell/hidden states.
        '''
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)

        if zm < 0. or zs > 1.:
            raise ValueError(
                'One/both provided Zoneout factors are not in [0, 1]')

        if not use_cudnn:
            self._cell = tf.nn.rnn_cell.LSTMCell(
                num_units, state_is_tuple=state_is_tuple, name=name)
        else:
            self._cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        self.is_training = is_training
        self.state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        '''Runs vanilla LSTM Cell and applies zoneout.
        '''
        # Apply vanilla LSTM
        output, new_state = self._cell(inputs, state, scope)

        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            num_proj = self._cell._num_units if self._cell._num_proj is None else self._cell._num_proj
            prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            prev_h = tf.slice(
                state, [0, self._cell._num_units], [-1, num_proj])
            new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
            new_h = tf.slice(
                new_state, [0, self._cell._num_units], [-1, num_proj])

        # Apply zoneout
        if self.is_training:
            # nn.dropout takes keep_prob (probability to keep activations) not drop_prob (probability to mask activations)!
            c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c -
                                                         prev_c, (1 - self._zoneout_cell), seed=global_seed) + prev_c
            h = (1 - self._zoneout_outputs) * tf.nn.dropout(new_h -
                                                            prev_h, (1 - self._zoneout_outputs), seed=global_seed) + prev_h

        else:
            c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
            h = (1 - self._zoneout_outputs) * \
                new_h + self._zoneout_outputs * prev_h

        new_state = tf.nn.rnn_cell.LSTMStateTuple(
            c, h) if self.state_is_tuple else tf.concat(1, [c, h])

        return output, new_state


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

def multihead_attention(queries,
                        keys,
			values,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0.,
                        is_training=True,
                        causality=False,
                        att_score='mul',
                        scope='multihead_attention',
                        reuse=None):
    '''Applies multihead attention.
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_k].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      values: A 3d tensor with shape of [N, T_k, C_v].
      num_units: A scalar. Attention size: output dim
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, use_bias=False)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, use_bias=False)  # (N, T_k, C)
        V = tf.layers.dense(values, num_units, use_bias=False)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        if att_score == 'mul':
            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5) # to overcome dot product magnitude problem

        elif att_score == 'cos':
            # Cosine Similarity: w * cosx + b
            w_cos = tf.get_variable(
                'cos_weight', shape=(), dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(seed=global_seed))
            b_cos = tf.get_variable(
                'cos_bias', shape=(), dtype=tf.float32,
                initializer=tf.zeros_initializer())
            outputs = tf.matmul(tf.nn.l2_normalize(Q_, -1), tf.transpose(tf.nn.l2_normalize(K_, -1), [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = w_cos * outputs + b_cos

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

	# pad -inf to the place where key is 0, for near zero softmax
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding, for Transformer decoder masked attention
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k), to_dense return the matrix, input: B1,...Bb,N,N, output: last two dim will be diagnol
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        alignment = outputs
        # Restore alignment shape
        alignment = tf.concat(tf.split(alignment, num_heads, axis=0), axis=2)  # [N,T_q,C]

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C), no need do this mask before softmax, since it is on T_k dimension. this is to eliminate all 0s attention query

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training), seed=global_seed)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # ADD & NORM
        # Residual connection
        # extra op ?
        # outputs += Q  # ?

        # Normalize
        # outputs = normalize(outputs)  # (N, T_q, C)

    return outputs, alignment

class Prenet:
    """Two fully connected layers used as an information bottleneck for the attention.
    """

    def __init__(self, is_training, layers_sizes=[256, 256], drop_rate=0.5, activation=tf.nn.relu, scope=None):
        """
        Args:
                layers_sizes: list of integers, the length of the list represents the number of pre-net
                        layers and the list values represent the layers number of units
                activation: callable, activation functions of the prenet layers.
                scope: Prenet scope.
        """
        super(Prenet, self).__init__()
        self.drop_rate = drop_rate

        self.layers_sizes = layers_sizes
        self.activation = activation
        self.is_training = is_training

        self.scope = 'prenet' if scope is None else scope

    def __call__(self, inputs):
        x = inputs

        with tf.variable_scope(self.scope):
            for i, size in enumerate(self.layers_sizes):
                dense = tf.layers.dense(x, units=size, activation=self.activation,
                                        name='dense_{}'.format(i + 1))
                # The paper discussed introducing diversity in generation at inference time
                # by using a dropout of 0.5 only in prenet layers (in both training and inference).
                x = tf.layers.dropout(dense, rate=self.drop_rate, training=True, seed=global_seed,
                                      name='dropout_{}'.format(i + 1) + self.scope)
        return x


class DecoderRNN:
    """Decoder two uni directional LSTM Cells
    """

    def __init__(self, is_training, layers=2, size=1024, zoneout=0.1, scope=None):
        """
        Args:
                is_training: Boolean, determines if the model is in training or inference to control zoneout
                layers: integer, the number of LSTM layers in the decoder
                size: integer, the number of LSTM units in each layer
                zoneout: the zoneout factor
        """
        super(DecoderRNN, self).__init__()
        self.is_training = is_training

        self.layers = layers
        self.size = size
        self.zoneout = zoneout
        self.scope = 'decoder_rnn' if scope is None else scope

        # Create a set of LSTM layers
        self.rnn_layers = [ZoneoutLSTMCell(size, is_training,
                                           zoneout_factor_cell=zoneout,
                                           zoneout_factor_output=zoneout,
                                           name='decoder_LSTM_{}'.format(i+1)) for i in range(layers)]

        self._cell = tf.contrib.rnn.MultiRNNCell(
            self.rnn_layers, state_is_tuple=True)

    def __call__(self, inputs, states):
        with tf.variable_scope(self.scope):
            return self._cell(inputs, states)


class FrameProjection:
    """Projection layer to r * num_mels dimensions or num_mels dimensions
    """

    def __init__(self, shape=80, activation=None, scope=None):
        """
        Args:
                shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
                activation: callable, activation function
                scope: FrameProjection scope.
        """
        super(FrameProjection, self).__init__()

        self.shape = shape
        self.activation = activation

        self.scope = 'Linear_projection' if scope is None else scope
        self.dense = tf.layers.Dense(
            units=shape, activation=activation, name='projection_{}'.format(self.scope))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            # If activation==None, this returns a simple Linear projection
            # else the projection will be passed through an activation function
            # output = tf.layers.dense(inputs, units=self.shape, activation=self.activation,
            # 	name='projection_{}'.format(self.scope))
            output = self.dense(inputs)

            return output

class Postnet:
    """Postnet that takes final decoder output and fine tunes it (using vision on past and future frames)
    """

    def __init__(self, is_training, hparams, activation=tf.nn.tanh, scope=None):
        """
        Args:
                is_training: Boolean, determines if the model is training or in inference to control dropout
                kernel_size: tuple or integer, The size of convolution kernels
                channels: integer, number of convolutional kernels
                activation: callable, postnet activation function for each convolutional layer
                scope: Postnet scope.
        """
        super(Postnet, self).__init__()
        self.is_training = is_training

        self.kernel_size = hparams.postnet_kernel_size
        self.channels = hparams.postnet_channels
        self.activation = activation
        self.scope = 'postnet_convolutions' if scope is None else scope
        self.postnet_num_layers = hparams.postnet_num_layers
        self.drop_rate = hparams.extractron_dropout_rate

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            x = inputs
            for i in range(self.postnet_num_layers - 1):
                x = conv1d(x, self.kernel_size, self.channels, self.activation,
                           self.is_training, self.drop_rate, 'conv_layer_{}_'.format(i + 1)+self.scope)
            x = conv1d(x, self.kernel_size, self.channels, lambda _: _, self.is_training, self.drop_rate,
                       'conv_layer_{}_'.format(5)+self.scope)
        return x


def conv1d(inputs, kernel_size, channels, activation, is_training, drop_rate, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=None,
            padding='same')
        batched = tf.layers.batch_normalization(
            conv1d_output, training=is_training)
        activated = activation(batched)
        return tf.layers.dropout(activated, rate=drop_rate, training=is_training, seed=global_seed,
                                 name='dropout_{}'.format(scope))


def _round_up_tf(x, multiple):
    # Tf version of remainder = x % multiple
    remainder = tf.mod(x, multiple)
    # Tf version of return x if remainder == 0 else x + multiple - remainder
    x_round = tf.cond(tf.equal(remainder, tf.zeros(tf.shape(remainder), dtype=tf.int32)),
                      lambda: x,
                      lambda: x + multiple - remainder)

    return x_round


def sequence_mask(lengths, r, expand=True):
    '''Returns a 2-D or 3-D tensorflow sequence mask depending on the argument 'expand'
    '''
    max_len = tf.reduce_max(lengths)
    max_len = _round_up_tf(max_len, tf.convert_to_tensor(r))
    if expand:
        return tf.expand_dims(tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32), axis=-1)
    return tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)

