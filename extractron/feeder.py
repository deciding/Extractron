import tensorflow as tf
from datasets.tfrecord import parse_single_preprocessed_data, decode_single_preprocessed_data

max_mel_len = 1000

class Feeder:
    """
            Feeds batches of data into queue on a background thread.
    """

    def __init__(self, hparams):
        super(Feeder, self).__init__()
        self._hparams = hparams

    def setup_dataset(self, dataset_path, eval_dataset_path):
        self.dataset=self.get_input_fn(dataset_path)()
        self.eval_dataset=self.get_input_fn(eval_dataset_path, False)()
        self.next=self.dataset.make_one_shot_iterator().get_next()
        self.eval_next=self.eval_dataset.make_one_shot_iterator().get_next()

    # here assume sorted on 32*32 batch size
    def get_input_fn(self, dataset_path, training=True):
        def input_fn():
            dataset = tf.data.TFRecordDataset(dataset_path)

            #if training:
            dataset = dataset.repeat()

            map_fn = self.get_map_fn(training)
            #filter_fn = self.get_filter_fn()
            #dataset = dataset.map(map_fn, num_parallel_calls=20)
            dataset = dataset.map(map_fn)
            #dataset = dataset.filter(filter_fn)

            dataset=dataset.batch(self._hparams.extractron_batch_size)
            #if training:
            #    dataset=dataset.batch(self._hparams.extractron_batch_size)
            #else:
            #    dataset=dataset.batch(1)

            #dataset.prefetch(4)
            #dataset = dataset.shuffle(self._hparams.shuffle_batch_size)
            return dataset

        return input_fn

    def get_map_fn(self, training=True):
        # target, mixed, speaker
        def map_fn(t):
            t=parse_single_preprocessed_data(t, need_phase=not training, need_mel=True)
            t=decode_single_preprocessed_data(t, self._hparams.num_freq, need_phase=not training, num_mels=80, need_mel=True)
            return t
        return map_fn

    def get_filter_fn(self):
        #text_length, target_length, text, mel, speaker
        def filter_fn(*t):
            return t[1] <= max_mel_len #tensor compare
        return filter_fn
