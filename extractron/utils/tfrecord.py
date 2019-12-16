# training structure
# context: (text_length, target_length)
# featurelist: (text, mel, linear, stop, speaker)

import tensorflow as tf
import numpy as np
from collections.abc import Iterable
from typing import List

def bytes_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_tfrecord(example: tf.train.Example, filename: str):
    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(example.SerializeToString())

def write_tfrecords(examples: List[tf.train.Example], filename: str):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for example in examples:
            writer.write(example.SerializeToString())

def convert_to_example(key: str, linear: np.ndarray, mel: np.ndarray, text: np.ndarray, speaker: np.ndarray):
    raw_linear = linear.tostring()
    raw_mel = mel.tostring()
    raw_text = text.tostring()
    raw_speaker = speaker.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'key': bytes_feature([key.encode('utf-8')]),
        'linear': bytes_feature([raw_linear]),
        'mel': bytes_feature([raw_mel]),
        'text': bytes_feature([raw_text]),
        'speaker': bytes_feature([raw_speaker]),
    }))
    return example

def parse_single_preprocessed_data(proto):
    features = {
        'key': tf.FixedLenFeature((), tf.string),
        'linear': tf.FixedLenFeature((), tf.string),
        'mel': tf.FixedLenFeature((), tf.string),
        'text': tf.FixedLenFeature((), tf.string),
        'speaker': tf.FixedLenFeature((), tf.string),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features

def decode_single_preprocessed_data(parsed, num_mels, num_freq):
    linear = tf.reshape(tf.decode_raw(parsed['linear'], tf.float32), [-1, num_freq])
    mel = tf.reshape(tf.decode_raw(parsed['mel'], tf.float32), [-1, num_mels])
    target_length = tf.shape(mel)[0]
    text = tf.decode_raw(parsed['text'], tf.int32)
    text_length = tf.shape(text)[0]
    speaker = tf.decode_raw(parsed['speaker'], tf.float32)
    #language_mask = np.asarray(
    #    list(seq_to_cnen_mask(text)), dtype=np.int32)

    #return PreprocessedData(
    #    key=parsed['key'],
    #    linear=linear,
    #    mel=mel,
    #    text=text,
    #    speaker=speaker,
    #)
    return text_length, target_length, text, mel, linear, speaker

