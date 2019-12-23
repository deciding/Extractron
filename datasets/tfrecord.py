# training structure
# context: (text_length, target_length)
# featurelist: (text, mel, linear, stop, speaker)

import tensorflow as tf
from collections.abc import Iterable
from typing import List
from tqdm import tqdm

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
        for example in tqdm(examples):
            writer.write(example.SerializeToString())

def convert_to_example(target, mixed, speaker,
        target_phase=None, mixed_phase=None, target_mel=None, mixed_mel=None):
    raw_target = target.tostring()
    raw_mixed = mixed.tostring()
    raw_speaker = speaker.tostring()
    raw_target_phase = target_phase.tostring() if target_phase is not None else None
    raw_mixed_phase = mixed_phase.tostring() if mixed_phase is not None else None
    raw_target_mel = target_mel.tostring() if target_mel is not None else None
    raw_mixed_mel = mixed_mel.tostring() if mixed_mel is not None else None
    #TODO: handle the case where only have phase info but no mel info
    if target_mel is not None and mixed_mel is not None:
        if target_phase is not None and mixed_phase is not None:
            example = tf.train.Example(features=tf.train.Features(feature={
                'target': bytes_feature([raw_target]),
                'mixed': bytes_feature([raw_mixed]),
                'speaker': bytes_feature([raw_speaker]),
                'target_phase': bytes_feature([raw_target_phase]),
                'mixed_phase': bytes_feature([raw_mixed_phase]),
                'target_mel': bytes_feature([raw_target_mel]),
                'mixed_mel': bytes_feature([raw_mixed_mel]),
            }))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'target': bytes_feature([raw_target]),
                'mixed': bytes_feature([raw_mixed]),
                'speaker': bytes_feature([raw_speaker]),
                'target_mel': bytes_feature([raw_target_mel]),
                'mixed_mel': bytes_feature([raw_mixed_mel]),
            }))
    else:
        example = tf.train.Example(features=tf.train.Features(feature={
            'target': bytes_feature([raw_target]),
            'mixed': bytes_feature([raw_mixed]),
            'speaker': bytes_feature([raw_speaker]),
        }))

    return example

def parse_single_preprocessed_data(proto, need_phase=False, need_mel=False):
    if need_phase:
        features = {
            'target': tf.FixedLenFeature((), tf.string),
            'mixed': tf.FixedLenFeature((), tf.string),
            'speaker': tf.FixedLenFeature((), tf.string),
            'target_phase': tf.FixedLenFeature((), tf.string),
            'mixed_phase': tf.FixedLenFeature((), tf.string),
            'target_mel': tf.FixedLenFeature((), tf.string),
            'mixed_mel': tf.FixedLenFeature((), tf.string),
        }
    elif need_mel:
        features = {
            'target': tf.FixedLenFeature((), tf.string),
            'mixed': tf.FixedLenFeature((), tf.string),
            'speaker': tf.FixedLenFeature((), tf.string),
            'target_mel': tf.FixedLenFeature((), tf.string),
            'mixed_mel': tf.FixedLenFeature((), tf.string),
        }
    else:
        features = {
            'target': tf.FixedLenFeature((), tf.string),
            'mixed': tf.FixedLenFeature((), tf.string),
            'speaker': tf.FixedLenFeature((), tf.string),
        }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features

def decode_single_preprocessed_data(parsed, spec_channels, need_phase=False, num_mels=80, need_mel=False):
    target = tf.reshape(tf.decode_raw(parsed['target'], tf.float32), [-1, spec_channels])
    mixed = tf.reshape(tf.decode_raw(parsed['mixed'], tf.float32), [-1, spec_channels])
    speaker = tf.decode_raw(parsed['speaker'], tf.float32)
    if need_phase:
        target_phase = tf.reshape(tf.decode_raw(parsed['target_phase'], tf.float32), [-1, spec_channels])
        mixed_phase = tf.reshape(tf.decode_raw(parsed['mixed_phase'], tf.float32), [-1, spec_channels])
        target_mel = tf.reshape(tf.decode_raw(parsed['target_mel'], tf.float32), [-1, num_mels])
        mixed_mel = tf.reshape(tf.decode_raw(parsed['mixed_mel'], tf.float32), [-1, num_mels])
        return target, mixed, target_phase, mixed_phase, target_mel, mixed_mel, speaker
    elif need_mel:
        target_mel = tf.reshape(tf.decode_raw(parsed['target_mel'], tf.float32), [-1, num_mels])
        mixed_mel = tf.reshape(tf.decode_raw(parsed['mixed_mel'], tf.float32), [-1, num_mels])
        return target, mixed, target_mel, mixed_mel, speaker
    else:
        return target, mixed, speaker
