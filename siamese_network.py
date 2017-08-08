# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
import os
import glob

def load_data(path, seq_list):
	input_data_list = []
	output_data_list = []
	for seq_no in seq_list:
		data = np.load(path+seq_no+".npy")
		input_data_list.append(data[:,0:2])
		output_data_list.append(data[:,2])
	input_data = np.concatenate(input_data_list,axis=0)
	input_data = np.asarray([[i[0], i[1]] for i in input_data])
	output_data = np.concatenate(output_data_list,axis=0)
	output_data = np.asarray([[i[0], i[1], i[2]] for i in output_data])
	return input_data, output_data

# def normalise(data):
# 	temp = np.zeros(shape=data.shape)
# 	temp[:,0] = (data[:,0]-128.0)/255.0
# 	temp[:,1] = (data[:,1]-128.0)/255.0
# 	# print temp[:,0]
# 	return temp

# def batch_normalise(batch):
# 	return (batch-128.0)/255.0

def shuffle(train_data, train_labels):

    permutation = np.random.permutation(train_labels.shape[0])
    shuffled_dataset = train_data[permutation]
    shuffled_labels = train_labels[permutation]
    return shuffled_dataset, shuffled_labels

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):

  # Get number of input channels
  input_channels = int(x.get_shape()[-1])

  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k,
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)

  with tf.variable_scope(name, reuse=None) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights',
                              shape = [filter_height, filter_width,
                              input_channels, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("bias", biases)

    # if groups == 1:
    conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights an
    # print conv.get_shape()
    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)

    return relu

def conv_(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):

  # Get number of input channels
  input_channels = int(x.get_shape()[-1])

  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k,
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)

  with tf.variable_scope(name, reuse=True) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights',
                              shape = [filter_height, filter_width,
                              input_channels, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("bias", biases)

    # if groups == 1:
    conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    # print conv.get_shape()
    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)

    return relu

def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name, reuse=None) as scope:

    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    biases = tf.get_variable('biases', [num_out], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("bias", biases)

    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)
      return relu
    else:
      return act

def fc2(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name, reuse=True) as scope:

    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', trainable=True)
    biases = tf.get_variable('biases', trainable=True)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("bias", biases)

    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)
      return relu
    else:
      return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x,
             name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
  return tf.nn.local_response_normalization(x, depth_radius = radius,
                                            alpha = alpha, beta = beta,
                                            bias = bias, name = name)

def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)




class Model():
	def __init__(self,  drop_prob, weights_path=None):
		self.not_trainable = ["conv1", "conv2", "conv3"]
		self.reinitialize = ["fc6", "fc7", "fc8"]
		# self.load_weights(weights_path)
		self.drop_prob = drop_prob
		

	def alexnet_1(self, image_batch):
		with tf.name_scope("cnn1") as scope:
			conv1 = conv(image_batch, 11, 11, 96, 4, 4, padding = 'SAME', name = 'conv1')
			# print conv1.get_shape()
			pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'SAME', name = 'pool1')
			norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')

			# 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
			conv2 = conv(norm1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
			pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'SAME', name ='pool2')
			norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')

			# 3rd Layer: Conv (w ReLu)
			conv3 = conv(norm2, 3, 3, 384, 1, 1, name = 'conv3')

			# 4th Layer: Conv (w ReLu) splitted into two groups
			conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')

			# 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
			conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
			pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'SAME', name = 'pool5')
			# dropout5 = dropout(pool5, self.drop_prob)

			# 6th Layer: Flatten -> FC (w ReLu) -> Dropout
			flattened = tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))])
			fc6 = fc(flattened, int(np.prod(pool5.get_shape()[1:])), 4096, name='fc6')

			return fc6

	def alexnet_2(self, next_image_batch):
		with tf.name_scope("cnn2") as scope:
			conv1 = conv_(next_image_batch, 11, 11, 96, 4, 4, padding = 'SAME', name = 'conv1')
			pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'SAME', name = 'pool1')
			norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')

			# 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
			conv2 = conv_(norm1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
			pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'SAME', name ='pool2')
			norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')

			# 3rd Layer: Conv (w ReLu)
			conv3 = conv_(norm2, 3, 3, 384, 1, 1, name = 'conv3')

			# 4th Layer: Conv (w ReLu) splitted into two groups
			conv4 = conv_(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')

			# 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
			conv5 = conv_(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
			pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'SAME', name = 'pool5')
			# dropout5 = dropout(pool5, self.drop_prob)
			# print "Dropout 5", dropout5.get_shape()
			# 6th Layer: Flatten -> FC (w ReLu) -> Dropout
			flattened = tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))])
			# print "Flattened", fla        ttened.get_shape()
			fc6 = fc2(flattened, int(np.prod(pool5.get_shape()[1:])), 4096, name='fc6')

			return fc6

	def siamese_net(self, feature_1, feature_2):
		with tf.name_scope("concatanation"):
			# print feature_1.get_shape(), feature_2.get_shape()
			concat_out = tf.concat([feature_1, feature_2],1, name="concat")
			# print concat_out.get_shape()
			# dropout_concat = dropout(concat_out, self.drop_prob)
			# concat_out has 8192 activations
			fc7 = fc(concat_out, 8192, 4096, name='fc7')
			dropout7 = dropout(fc7, self.drop_prob)
			# fc7 = fc(dropout_concat, 8192, 3, name='fc7')

			fc8 = fc(dropout7, 4096, 256, name='fc8')
			fc9 = fc(fc8, 256, 3, name='fc9')

			return fc9

	def load_weights(self, weights_path):
		if len(glob.glob("./checkpoints/model*")) != 0:
			print "Will Use saved weights......."
		
		else:
			print "Using Pretrained weights."
			weights_dict = np.load(weights_path, encoding = 'bytes').item()
			for op_name in weights_dict:
			# Check if the layer is one of the layers that should be reinitialized
				if op_name in self.not_trainable:
					with tf.variable_scope(op_name, reuse = None):
					 # Loop over list of weights/biases and assign them to their corresponding tf variable
						for data in weights_dict[op_name]:
						  	# Biases
							if len(data.shape) == 1:
								var = tf.get_variable('biases',shape = data.shape, trainable = False)
								sess.run(var.assign(data))

						          # Weights
							else:
								var = tf.get_variable('weights',shape = data.shape, trainable = False)
								sess.run(var.assign(data))
				else:
					if op_name not in self.reinitialize:
						with tf.variable_scope(op_name, reuse = None):
						 # Loop over list of weights/biases and assign them to their corresponding tf variable
							for data in weights_dict[op_name]:
					          # Biases
								if len(data.shape) == 1:
									var = tf.get_variable('biases',shape = data.shape, trainable = True)
									sess.run(var.assign(data))
							  # Weights
								else:
									var = tf.get_variable('weights',shape = data.shape, trainable = True)
									sess.run(var.assign(data))


	def train_model(self, input_data, output_data, val_input, val_output, leanring_rate, no_epochs, batch_size, beta):
		""" This trains model and saves its weight after each epoch"""

		with tf.device("/gpu:1"):
			with tf.name_scope("Input") as scope:
				x_image = tf.placeholder(tf.float32, [batch_size, 188, 620, 3])
				x_next_image = tf.placeholder(tf.float32, [batch_size, 188, 620, 3])
				y = tf.placeholder(tf.float32, [batch_size, 3])

			with tf.name_scope("Network") as scope:
				feature_1 = self.alexnet_1(x_image)
				feature_2 = self.alexnet_2(x_next_image)
				final_output = self.siamese_net(feature_1, feature_2)

			# with tf.variable_scope(tf.get_variable_scope(), reuse=True) as  scope:
			# 	weight_conv1 = tf.get_variable("conv1/weights")
			# 	weight_conv2 = tf.get_variable("conv2/weights")
			# 	weight_conv3 = tf.get_variable("conv3/weights")
			# 	weight_conv4 = tf.get_variable("conv4/weights")
			# 	weight_conv5 = tf.get_variable("conv5/weights")
			# 	weight_fc6 = tf.get_variable("fc6/weights")
			# 	weight_fc7 = tf.get_variable("fc7/weights")
			# 	weight_fc8 = tf.get_variable("fc8/weights")

			with tf.name_scope("Training") as scope:
				# regu_loss = tf.nn.l2_loss(weight_conv1) + tf.nn.l2_loss(weight_conv2) + tf.nn.l2_loss(weight_conv3) + tf.nn.l2_loss(weight_conv4)+\
				# 			tf.nn.l2_loss(weight_conv5) + tf.nn.l2_loss(weight_fc6) + tf.nn.l2_loss(weight_fc7) + tf.nn.l2_loss(weight_fc8)

				data_loss = tf.reduce_mean(tf.square(tf.subtract(final_output,y)))

				# loss = data_loss + beta * regu_loss
				loss = data_loss

				train_op = tf.train.AdamOptimizer(leanring_rate).minimize(loss)
			
			with tf.name_scope("Loss_Monitoring") as scope:
				train_sum_op = tf.summary.scalar("Training Loss", loss)
				merged_sum = tf.summary.merge_all()
				# val_sum_op = tf.summary.scalar("Vaildation", loss)

			if len(glob.glob("./checkpoints/model*")) != 0:
				tf.reset_default_graph()
				saver = tf.train.import_meta_graph("./checkpoints/model.meta")
				saver.restore(sess,'./checkpoints/model')
				print "Restoring saved weights....."
			else:
				config = tf.ConfigProto(allow_soft_placement=True)
				config.gpu_options.allow_growth = True
				sess = tf.Session(config=config)
				sess.run(tf.global_variables_initializer())
				saver = tf.train.Saver()
				print "Initializing variables, no saved weights found ...."
			
			writer = tf.summary.FileWriter("./logs/graphs")
			writer.add_graph(sess.graph)

			# total_parameters = 0
			for variable in tf.trainable_variables():
	  #   	# shape is an array of tf.Dimension
				shape = variable.name
				print(shape)
			#     print(len(shape))
			#     variable_parametes = 1
			#     for dim in shape:
			#         print(dim)
			#         variable_parametes *= dim.value
			#     print(variable_parametes)
			#     total_parameters += variable_parametes
			# print(total_parameters)

			for epoch in range(no_epochs):
				input_data, output_data = shuffle(input_data, output_data)
				eval_data, eval_labels = shuffle(val_input, val_output)
				for k in xrange(0, len(input_data), batch_size):
					batch_input = input_data[k:k+batch_size]
					batch_output = output_data[k:k+batch_size]
					batch_input = np.asarray([[i[0], i[1]] for i in batch_input])
					batch_output = np.asarray([[i[0], i[1], i[2]] for i in batch_output])
					# print batch_output[0]
					# cv2.imshow("Input_Image", batch_input[0,0])
					# cv2.imshow("Input_Image_Next", batch_input[0,1])
					# cv2.waitKey(100)
					# result, feature1, feature2, _ , summary = sess.run([final_output,feature_1, feature_2, train_op, merged_sum],{x_image:batch_input[:,0], x_next_image:batch_input[:,1], y:batch_ouput})
					_ , summary = sess.run([train_op, merged_sum],{x_image:batch_input[:,0], x_next_image:batch_input[:,1], y:batch_output})
					writer.add_summary(summary)
					# print "Results", result
					# print "Feature_1", feature1
					# print "Feature_2", feature2
					# print "Epoch:-", epoch,"Sequence:-",seq_no,"Batch Number:-", k/batch_size+1

					itr = epoch*len(input_data)/batch_size + k/batch_size
					if (itr % 100 == 0):
						writer.add_summary(summary, itr) # append summary     
						print "Epoch:-", epoch, "Iteration:-", k/batch_size  
				
				# val_loss_tot = 0
				# for j in xrange(0, len(val_input)-batch_size, batch_size):
				# 	val_batch_input = val_input[j:j+batch_size]
				# 	val_batch_output = val_output[j:j+batch_size]
										
				# 	val_loss_tot = val_loss_tot + sess.run(loss, {x_image:val_batch_input[:,0], x_next_image:val_batch_input[:,1], y:val_batch_output})

				# val_loss_avg = val_loss_tot/int(len(val_input)/batch_size)
				# train_loss = sess.run(loss, {x_image:batch_input[:,0], x_next_image:batch_input[:,1], y:batch_ouput})
				# print "Validation Loss:", val_loss_avg
				# print "Training Loss:", train_loss
				l2_loss_avg = 0
				for mark in xrange(0, len(eval_labels), batch_size):   
					batch_input, batch_output = eval_data[mark : mark + batch_size] , eval_labels[mark : mark + batch_size] 
					batch_input = np.asarray([[i[0], i[1]] for i in batch_input])
					# batch_input = normalize(batch_input)
					# batch_input = batch_input[:, :, :, :, np.newaxis]
					batch_output = np.asarray([[i[0], i[1], i[2]] for i in batch_output])
					#_, eval_summary = sess.run([l2_loss, eval_summ], feed_dict = {x_image_t: batch_input[:,0], x_image_t_1: batch_input[:,1], target: batch_output, drop7_prob: 1.0 })    
					l2_loss_avg = l2_loss_avg + sess.run(l2_loss, feed_dict = {x_image: batch_input[:, 0], x_next_image: batch_input[:,1], y: batch_output})

				l2_loss_avg = l2_loss_avg/((int)(len(eval_labels))/batch_size)
				eval_summ = sess.run(tf.summary.scalar("validation_loss", l2_loss_avg))
				writer.add_summary(eval_summ, itr)
				print "Validation Loss Updated."

				if (epoch % 5 == 0): 
					saver.save(sess, "checkpoints/model")
					print "Checkpoint saved"

	def predict(self, image=None, next_image=None):
		
		feature_1 = self.alexnet_1(image)
		feature_2 = self.alexnet_2(next_image)
		final_ouput = self.siamese_net()
		return final_ouput


data_path = "./training_testing_data/620x188_images/"
# train_seq = ["02", "00", "05", "06", "07", "01", "09", "10","08","04"]
# val_seq = ["03"]
# val_input, val_output = load_data(data_path, val_seq)
# val_input, val_output = val_input[0:1000], val_output[0:1000]
# train_input, train_output = load_data(data_path, train_seq)
train_full = np.load(data_path+'training_1.npy')
# print train_full.shape
train_input = train_full[:, 0:2]
train_output = train_full[:, 2]
# print train_data.shape, train_labels.shape

eval_full = np.load(data_path+'val_1.npy')
val_input = eval_full[:, 0:2]
val_output = eval_full[:, 2]
print train_input.shape, train_output.shape
# val_input = np.random.randn(12,2,126,414,3)
# val_output = np.random.randn(12,3)
# with tf.Session() as sess:
siam_model = Model(drop_prob= 0.5, weights_path="./bvlc_alexnet.npy")
siam_model.train_model(input_data=train_input, output_data = train_output,val_input=val_input, val_output=val_output, leanring_rate=0.00001, no_epochs=80, batch_size=4, beta=0.5)