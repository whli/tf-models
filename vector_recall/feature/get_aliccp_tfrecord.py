#!/usr/bin/env python
#coding=utf-8
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
sys.path.append("../")

import tensorflow as tf
import numpy as np
import re
from multiprocessing import Pool as ThreadPool

import config.config as config

input_dir = config.DATA_LIB
output_dir = config.DATA_TFRECORD
threads = 1
Common_Fileds = config.COMMAN_FIELDS
Multi_Fields = config.MULTI_FIELDS

# 0,216:9342395:1.0 301:9351665:1.0 205:7702673:1.0 206:8317829:1.0 207:8967741:1.0 508:9356012:2.30259 
# 210:9059239:1.0 210:9042796:1.0 210:9076972:1.0 210:9103884:1.0 210:9063064:1.0 
# 127_14:3529789:2.3979 127_14:3806412:2.70805  => one field multi feature
def gen_tfrecords(in_file):
    basename = os.path.basename(in_file) + ".tfrecord"
    out_file = os.path.join(output_dir, basename)
    tfrecord_out = tf.python_io.TFRecordWriter(out_file)
    with open(in_file) as fi:
        for line in fi:
            fields = line.strip().split(',')
            if len(fields) != 2:
                continue
            #1 label
            y = [float(fields[0])]
            feature = {
                "y": tf.train.Feature(float_list = tf.train.FloatList(value=y))
             }

            splits = re.split('[ :]', fields[1])
            ffv = np.reshape(splits,(-1,3))
            #2 不需要特殊处理的特征
            feat_ids = np.array([])
            feat_vals = np.array([])
            for f, def_id in Common_Fileds.iteritems():
                if f in ffv[:,0]:   # field have then feature index , other field_index
                    mask = np.array(f == ffv[:,0])
                    feat_ids = np.append(feat_ids, ffv[mask,1])
                    feat_vals = np.append(feat_vals,ffv[mask,2].astype(np.float))
                else:
                    feat_ids = np.append(feat_ids, def_id)
                    feat_vals = np.append(feat_vals,1.0)
            feature.update({"feat_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int))),
                            "feat_vals": tf.train.Feature(float_list=tf.train.FloatList(value=feat_vals))})

            #3 特殊字段单独处理
            '''
            for f, (fname, def_id) in Multi_Fileds.iteritems():
                if f in ffv[:,0]:
                    mask = np.array(f == ffv[:,0])
                    feat_ids = ffv[mask,1]
                    feat_vals= ffv[mask,2]
                else:
                    feat_ids = np.array([def_id])
                    feat_vals = np.array([1.0])
                # add name flg
                feature.update({fname+"ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int))),
                                fname+"vals": tf.train.Feature(float_list=tf.train.FloatList(value=feat_vals.astype(np.float)))})
            '''
            for f, (fname, def_id) in Multi_Fields.iteritems():    # only have ids
                if f in ffv[:,0]:
                    mask = np.array(f == ffv[:,0])
                    feat_ids = ffv[mask,1]
                else:
                    feat_ids = np.array([def_id])
                feature.update({fname+"ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int)))})

            # serialized to Example
            example = tf.train.Example(features = tf.train.Features(feature = feature))
            serialized = example.SerializeToString()
            tfrecord_out.write(serialized)
    tfrecord_out.close()

def main(_):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    #file_list = glob.glob(os.path.join(input_dir, "*-*"))
    file_list = [input_dir]

    pool = ThreadPool(threads) # Sets the pool size
    pool.map(gen_tfrecords, file_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
