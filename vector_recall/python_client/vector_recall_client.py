#coding=utf-8
	
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import time

tf.app.flags.DEFINE_string('server', 'localhost:9000',
						   'Server host:port.')
tf.app.flags.DEFINE_string('model', 'vec_recall',
						   'Model name.')
tf.app.flags.DEFINE_integer("common_field_size", 14, "Number of common fields")
FLAGS = tf.app.flags.FLAGS

def input_fn(filenames):
	def _parse_fn(record):
		features = { 
			"y": tf.FixedLenFeature([], tf.float32),
			"feat_ids": tf.FixedLenFeature([FLAGS.common_field_size], tf.int64),	   # for_change common field size ??
			"feat_vals": tf.FixedLenFeature([FLAGS.common_field_size], tf.float32),
			"videoIdsids": tf.VarLenFeature(tf.int64),
		}	
		parsed = tf.parse_single_example(record, features)
		y = parsed.pop('y')
		return parsed

	# Extract lines from input files using the Dataset API, can pass one filename or filename list
	dataset = tf.data.TFRecordDataset(filenames)#.map(_parse_fn) #, num_parallel_calls=10).prefetch(500000)	  # multi-thread pre-process then prefetch
	dataset = dataset.batch(10)
	iterator = dataset.make_one_shot_iterator()
	return iterator

def main():
	host, port = FLAGS.server.split(':')
	channel = implementations.insecure_channel(host, int(port))
	stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

	test_file = ["/tensorflow-serving/vec_model/libtest.tfrecord"]
	iterator = input_fn(test_file)
	serialized_example = iterator.get_next()
	with tf.Session() as session:
		for i in range(2):
			ser_list = session.run([serialized_example])
			ser_list = np.reshape(ser_list,(10))

			request = predict_pb2.PredictRequest()
			request.model_spec.name = FLAGS.model
			request.model_spec.signature_name = 'serving_default'

			request.inputs['input'].CopyFrom(
				tf.contrib.util.make_tensor_proto(ser_list))

			#result_future = stub.Predict.future(request, 10.0)
			result_future = stub.Predict(request, 10.0)
			print(result_future.outputs['output'])

if __name__ == "__main__":
	main()
