import os
import tensorflow as tf
import torch
from tqdm import tqdm
from glob import glob
import numpy as np
from utils.hparams import HParam
from datasets.tfrecord import convert_to_example

#python tfrecord_producer.py --in_dir training_libri/train/ --out xyz --gpu 6

class TFRecordProducer:
    def remove_list(self, list1, list2):
        i,j=0,0
        tmp_list1=[]
        tmp_list2=[]
        while i<len(list1) and j<len(list2):
            item1=int(list1[i].split('/')[-1].split('-')[0])
            item2=int(list2[j].split('/')[-1].split('-')[0])
            if item1==item2:
                tmp_list1.append(list1[i])
                tmp_list2.append(list2[j])
                i+=1
                j+=1
            elif item1<item2:
                i+=1
            else:
                j+=1
        return tmp_list1, tmp_list2

    def __init__(self, in_dir, hp, args, is_train):
        def find_all(file_format):
            return sorted(glob(os.path.join(self.data_dir, file_format)))

        self.in_dir=in_dir
        self.hp = hp
        self.args = args
        self.is_train = is_train
        #self.data_dir = hp.data.train_dir if is_train else hp.data.test_dir
        self.data_dir = in_dir

        self.dvec_list = find_all(hp.form.dvec_npy)
        self.target_wav_list = find_all(hp.form.target.wav)
        self.mixed_wav_list = find_all(hp.form.mixed.wav)
        self.target_mag_list = find_all(hp.form.target.mag)
        self.mixed_mag_list = find_all(hp.form.mixed.mag)
        self.target_phase_list = find_all(hp.form.target.phase)
        self.mixed_phase_list = find_all(hp.form.mixed.phase)
        _, self.target_wav_list=self.remove_list(self.dvec_list, self.target_wav_list)
        _, self.mixed_wav_list=self.remove_list(self.dvec_list, self.mixed_wav_list)
        _, self.target_mag_list=self.remove_list(self.dvec_list, self.target_mag_list)
        _, self.mixed_mag_list=self.remove_list(self.dvec_list, self.mixed_mag_list)
        _, self.target_phase_list=self.remove_list(self.dvec_list, self.target_phase_list)
        _, self.mixed_phase_list=self.remove_list(self.dvec_list, self.mixed_phase_list)

        print(len(self.dvec_list), len(self.target_wav_list), len(self.mixed_wav_list), \
            len(self.target_mag_list), len(self.mixed_mag_list),
            len(self.target_phase_list), len(self.mixed_phase_list))
        assert len(self.dvec_list) == len(self.target_wav_list) == len(self.mixed_wav_list) == \
            len(self.target_mag_list) == len(self.mixed_mag_list), "number of training files must match"
        assert len(self.dvec_list) != 0, \
            "no training file found"

    def write_training_examples_to_tfrecords(self, filename, need_phase=False):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for i in tqdm(range(len(self.dvec_list))):
                target_mag = torch.load(self.target_mag_list[i]).numpy()
                mixed_mag = torch.load(self.mixed_mag_list[i]).numpy()
                dvec = np.load(self.dvec_list[i])
                if need_phase:
                    target_phase = torch.load(self.target_phase_list[i]).numpy()
                    mixed_phase = torch.load(self.mixed_phase_list[i]).numpy()
                    writer.write(convert_to_example(target_mag, mixed_mag, dvec, target_phase, mixed_phase).SerializeToString())
                else:
                    writer.write(convert_to_example(target_mag, mixed_mag, dvec).SerializeToString())


import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--in_dir', default='.', help='input glob str')
parser.add_argument('-c', '--config', type=str, default='config/config.yaml',
                    help="yaml file for configuration")
parser.add_argument('--out', default='.', help='output prefix of tf record')
parser.add_argument('--gpu', default=0,
                    help='Path to model checkpoint')
parser.add_argument('--need_phase', action='store_true')
args=parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

in_dir=args.in_dir
filename=args.out
hp = HParam(args.config)
tp=TFRecordProducer(in_dir, hp, args, True)
tp.write_training_examples_to_tfrecords(filename, args.need_phase)

