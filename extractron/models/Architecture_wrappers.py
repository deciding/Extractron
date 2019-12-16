"""A set of wrappers usefull for extractron architecture
All notations and variable names were used in concordance with originial tensorflow implementation
"""
import collections

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

class ExtractronDecoderCellState(
        collections.namedtuple("ExtractronDecoderCellState",
                               ("cell_state", "time",
                                ))):

    def replace(self, **kwargs):
        """Clones the current state while overwriting components provided by kwargs.
        """
        return super(ExtractronDecoderCellState, self)._replace(**kwargs)


class ExtractronDecoderCell(RNNCell):
    """Tactron 2 Decoder Cell
    Decodes encoder output and previous mel frames into next r frames

    Decoder Step i:
            1) Prenet to compress last output information
            2) Concat compressed inputs with previous context vector (input feeding) *
            3) Decoder RNN (actual decoding) to predict current state s_{i} *
            4) Compute new context vector c_{i} based on s_{i} and a cumulative sum of previous alignments *
            5) Predict new output y_{i} using s_{i} and c_{i} (concatenated)
            6) Predict <stop_token> output ys_{i} using s_{i} and c_{i} (concatenated)

    * : This is typically taking a vanilla LSTM, wrapping it using tensorflow's attention wrapper,
    and wrap that with the prenet before doing an input feeding, and with the prediction layer
    that uses RNN states to project on output space. Actions marked with (*) can be replaced with
    tensorflow's attention wrapper call if it was using cumulative alignments instead of previous alignments only.
    """

    def __init__(self, prenet, rnn_cell, frame_projection):
        """Initialize decoder parameters

        Args:
            prenet: A tensorflow fully connected layer acting as the decoder pre-net
            attention_mechanism: A _BaseAttentionMechanism instance, usefull to
                    learn encoder-decoder alignments
            rnn_cell: Instance of RNNCell, main body of the decoder
            frame_projection: tensorflow fully connected layer with r * num_mels output units
            stop_projection: tensorflow fully connected layer, expected to project to a scalar
                    and through a sigmoid activation
                mask_finished: Boolean, Whether to mask decoder frames after the <stop_token>
        """
        super(ExtractronDecoderCell, self).__init__()
        # Initialize decoder layers
        self._prenet = prenet
        self._cell = rnn_cell
        self._frame_projection = frame_projection

    @property
    def output_size(self):
        return self._frame_projection.shape

    @property
    def state_size(self):
        """The `state_size` property of `ExtractronDecoderCell`.

        Returns:
          An `ExtractronDecoderCell` tuple containing shapes used by this object.
        """
        return ExtractronDecoderCellState(
            cell_state=self._cell._cell.state_size,
            time=tensor_shape.TensorShape([]),
            )

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.

        Args:
          batch_size: `0D` integer tensor: the batch size.
          dtype: The internal state data type.
        Returns:
          An `ExtractronDecoderCellState` tuple containing zeroed out tensors and,
          possibly, empty `TensorArray` objects.
        Raises:
          ValueError: (or, possibly at runtime, InvalidArgument), if
                `batch_size` does not match the output size of the encoder passed
                to the wrapper object at initialization time.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            cell_state = self._cell._cell.zero_state(batch_size, dtype)
            cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)
            return ExtractronDecoderCellState(
                cell_state=cell_state,
                time=array_ops.zeros([], dtype=tf.int32))

    def __call__(self, inputs, state):
        # Information bottleneck (essential for learning attention)
        prenet_output = self._prenet(inputs)


        # Unidirectional LSTM layers
        LSTM_output, next_cell_state = self._cell(prenet_output, state.cell_state)

        cell_outputs = self._frame_projection(LSTM_output)


        # Prepare next decoder state
        next_state = ExtractronDecoderCellState(
            time=state.time + 1,
            cell_state=next_cell_state)

        return cell_outputs, next_state
