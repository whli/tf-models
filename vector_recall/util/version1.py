#!/usr/bin/env python
#coding=utf-8

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import argparse
import shutil
import sys
import os
import json
import glob
from datetime import date, timedelta
from time import time

import math
import random
import tensorflow as tf

# opti: optimizer,deep_layers,dropout,learning_rate,batch_size,num_epochs,embedding_size

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.app.flags.DEFINE_integer("feature_size", 21861, "Number of features")
tf.app.flags.DEFINE_integer("common_field_size", 14, "Number of common fields")
tf.app.flags.DEFINE_integer("field_size", 15, "Number of fields")        
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 32, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'sgd', "optimizer type {adam, adagrad, sgd, momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("attention_pooling", True, "attention pooling")
tf.app.flags.DEFINE_string("attention_layers", '256', "Attention Net mlp layers")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", '../checkpoint', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '../servering', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("tensorboard_dir", "../tensorboard", "")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer_user, infer_video, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")
tf.app.flags.DEFINE_integer("cate_size", 5000, "cate num")

def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    def _parse_fn(record):
        features = {
            "y": tf.FixedLenFeature([], tf.float32),
            "feat_ids": tf.FixedLenFeature([FLAGS.common_field_size], tf.int64),       # for_change common field size ??
            "feat_vals": tf.FixedLenFeature([FLAGS.common_field_size], tf.float32),
            "videoIdsids": tf.VarLenFeature(tf.int64),
        }
        parsed = tf.parse_single_example(record, features)
        y = parsed.pop('y')
        return parsed, y

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TFRecordDataset(filenames).map(_parse_fn, num_parallel_calls=10).prefetch(500000)    # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    #return dataset.make_one_shot_iterator()
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def dnn_layer(features, labels, params):
    #------hyperparameters----
    cate_size = params["cate_size"]
    field_size = params["field_size"]
    common_field_size = params["common_field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    layers = map(int, params["deep_layers"].split(','))
    dropout = map(float, params["dropout"].split(','))
    common_dims = common_field_size*embedding_size

    #------bulid weights------
    Feat_Emb = tf.get_variable(name='embeddings', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feaure-------
    #{U-A-X-C不需要特殊处理的特征}
    feat_ids    = features['feat_ids']
    feat_vals   = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, common_field_size, 1])
    #{multi-hot}
    video_ids   = features['videoIdsids'] 

    #------build f(x)------
    with tf.variable_scope("Embedding-layer"):
        common_embs = tf.nn.embedding_lookup(Feat_Emb, feat_ids)                # None * F' * K
        uac_emb     = tf.multiply(common_embs, feat_vals)
        video_emb   = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=video_ids, sp_weights=None, combiner="sum")

    with tf.variable_scope("DNN-layer"):
        if FLAGS.batch_norm:
            if FLAGS.task_type == 'train':
                train_phase = True
            else:
                train_phase = False
        else:
            normalizer_fn = None
            normalizer_params = None

        x_deep = tf.concat([tf.reshape(uac_emb,shape=[-1, common_dims]),video_emb],axis=1) # None * (F*K)
        for i in range(len(layers)):
            x_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=layers[i], weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='dnn%d' % i)
            if FLAGS.batch_norm:
                x_deep = batch_norm_layer(x_deep, train_phase=train_phase, scope_bn='bn_%d' %i)   
            if FLAGS.task_type == 'train':
                x_deep = tf.nn.dropout(x_deep, keep_prob=dropout[i])                              #Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
    vector_size = layers[-1]
    with tf.variable_scope("weights"):
      nce_weights = tf.Variable(
          tf.truncated_normal(
              [cate_size, vector_size],
              stddev=1.0 / math.sqrt(vector_size)))
    with tf.name_scope('biases'):
      nce_biases = tf.Variable(tf.zeros([cate_size]))   

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=labels,
                inputs=x_deep,
                num_sampled=5,
                num_classes=cate_size))
    return loss,x_deep,nce_weights

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z

def restore_from_checkpoint(sess, saver, checkpoint):
  if checkpoint:
    print("Restore session from checkpoint: {}".format(checkpoint))
    saver.restore(sess, checkpoint)
    return True
  else:
    print("Checkpoint not found: {}".format(checkpoint))
    return False

def get_optimizer(learning_rate,optimizer_name):
  if optimizer_name == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer_name == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif optimizer_name == "adagrad":
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  elif optimizer_name == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif optimizer_name == "ftrl":
    optimizer = tf.train.FtrlOptimizer(learning_rate)
  elif optimizer_name == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
  else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  return optimizer


def main(_):
    #------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')

    checkpoint_file = FLAGS.model_dir + "/checkpoint.ckpt"
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)

    #------init Envs------
    train_data = "/mnt1/local_v1/vector_recall/train_data/lib.tfrecord"
    tr_files = [train_data]
    #tr_files = glob.glob("%s/tr/*tfrecord" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    #------bulid Tasks------
    model_params = {
        "cate_size":FLAGS.cate_size,   
        "common_field_size":FLAGS.common_field_size,
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout,
        "attention_layers": FLAGS.attention_layers
    }

    features,labels = input_fn(tr_files,FLAGS.batch_size)
    labels = tf.reshape(labels, shape = [-1,1])
    loss,x_deep,nce_weights = dnn_layer(features, labels, model_params)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = get_optimizer(FLAGS.learning_rate,FLAGS.optimizer)
    optimizer_op = optimizer.minimize(loss,global_step=global_step)
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        writer = tf.summary.FileWriter(FLAGS.tensorboard_dir, session.graph)
        init.run()
        check_flg = restore_from_checkpoint(session, saver, latest_checkpoint)
        if FLAGS.task_type == "train":
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=session)
            try:
                while not coord.should_stop():
                    run_metadata = tf.RunMetadata()
                    _,step = session.run([optimizer_op,global_step],run_metadata=run_metadata)
                    #if step == (num_steps - 1):
                    #    writer.add_run_metadata(run_metadata, 'step%d' % step)
                    if step % 100 == 0:
                        summary, loss_val = session.run([merged, loss])
                        writer.add_summary(summary, step)
                        print("step %s loss %s" %(step,loss_val))
                        saver.save(session, checkpoint_file, global_step=step)
            except tf.errors.OutOfRangeError:
                print("Data is cover!")
            coord.request_stop()
            coord.join(threads)
        elif FLAGS.task_type == "infer_user":
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=session)
            try:
                while not coord.should_stop():
                    x_deep_val = session.run([x_deep])
                    print(x_deep_val)
            except tf.errors.OutOfRangeError:
                print("Data is cover!")
            coord.request_stop()
            coord.join(threads)
        elif FLAGS.task_type == "infer_video":
            video_vec = session.run([nce_weights])
            for vec in video_vec[0]:
                print(vec)
        elif FLAGS.task_type == "export":
            server_version = int(time())
            server_path = FLAGS.servable_model_dir + "/" + str(server_version)
            builder = tf.saved_model.builder.SavedModelBuilder(server_path)
            inputs = {
                        'feat_ids': tf.saved_model.utils.build_tensor_info(tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.common_field_size])),
                        'feat_vals': tf.saved_model.utils.build_tensor_info(tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.common_field_size])),
                        "videoIdsids": tf.saved_model.utils.build_tensor_info(tf.placeholder(tf.int64))
                    }
            output = {'output': tf.saved_model.utils.build_tensor_info(x_deep)}
            prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
               inputs=inputs,
               outputs=output,
               method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
            builder.add_meta_graph_and_variables(
                session,
                [tf.saved_model.tag_constants.SERVING],
                {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}
            )
            builder.save()
        else:
            pass

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
