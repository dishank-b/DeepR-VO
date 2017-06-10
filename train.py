# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import tensorflow as tf
import os, csv
from PIL import Image


data_path = "/home/deeplearning/work/py-faster-rcnn/VO_2.2/dataset/training_npy"
result_path = "/home/deeplearning/work/py-faster-rcnn/VO_2.2/results/output"
log_path = "/home/deeplearning/work/py-faster-rcnn/VO_2.2/tensorflow_logs/6"

def check_data_img(nparray):
	img = Image.fromarray(nparray.reshape([256, 256, 3]), 'RGB')
	img.show()
	#img.close()

def weight_variable(shape, name):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    #return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    # initial = tf.constant(0.1, shape=shape)
    # return tf.Variable(initial, name=name)
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def weight_variable_conv2d(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

# def bias_variable_conv2d(shape, name):
#     return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())    

def separation(shuffled_dataset, shuffled_labels, n_validate = 0):
    
    train_data = shuffled_dataset[n_validate:]
    train_labels = shuffled_labels[n_validate:]

    valid_data = shuffled_dataset[:n_validate]
    valid_labels = shuffled_labels[:n_validate]

    return train_data, train_labels, valid_data, valid_labels

def shuffle(train_data, train_labels):

    permutation = np.random.permutation(train_labels.shape[0])
    shuffled_dataset = train_data[permutation]
    shuffled_labels = train_labels[permutation]
    return shuffled_dataset, shuffled_labels

def normalize(data):
    normalized_data = (data.astype(np.float32) - 128.0) / 255.0
    return normalized_data

def siamese_network_model(x_image):  

    with tf.name_scope("conv1"):
        w_conv1 = weight_variable_conv2d([11, 11, 1, 48], "w_conv1")
        b_conv1 = bias_variable([48], "b_conv1")
        conv1 = tf.nn.conv2d(x_image, w_conv1, strides=[1, 4, 4, 1], padding='SAME', name="conv1") + b_conv1
    
    with tf.name_scope("relu1"):
        relu1 = tf.nn.relu(conv1, name="relu1")
    
    with tf.name_scope("norm1"):
        norm1 = tf.nn.lrn(relu1, alpha=0.0001, beta=0.75, name="norm1")

    with tf.name_scope("pool1"):
        pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

    with tf.name_scope("conv2"):
        w_conv2 = weight_variable_conv2d([5, 5, 48, 128], "w_conv2")
        b_conv2 = bias_variable([128], "b_conv2")
        padding = [[0,0],[2,2],[2,2],[0,0]]
        pad1 = tf.pad(pool1, padding, "CONSTANT")
        conv2 = tf.nn.conv2d(pad1, w_conv2, strides=[1, 1, 1, 1], padding= 'SAME', name="conv2") + b_conv2
    
    with tf.name_scope("relu2"):
        relu2 = tf.nn.relu(conv2, name="relu2")

    with tf.name_scope("norm2"):
        norm2 = tf.nn.lrn(relu2, alpha=0.0001, beta=0.75, name="norm2")

    with tf.name_scope("pool2"):    
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")

    with tf.name_scope("conv3"): 
        w_conv3 = weight_variable_conv2d([3, 3, 128, 192], "w_conv3")
        b_conv3 = bias_variable([192], "b_conv3")
        padding = [[0,0], [1,1], [1,1], [0,0]]
        pad2 = tf.pad(pool2, padding, "CONSTANT")
        conv3 = tf.nn.conv2d(pad2, w_conv3, strides=[1, 1, 1, 1], padding='SAME', name="conv3") + b_conv3
    
    with tf.name_scope("relu3"):
        relu3 = tf.nn.relu(conv3, name="relu3")

    with tf.name_scope("conv4"):
        w_conv4 = weight_variable_conv2d([3, 3, 192, 192], "w_conv4")
        b_conv4 = bias_variable([192], "b_conv4")
        padding = [[0,0], [1,1], [1,1], [0,0]]
        pad3 = tf.pad(relu3, padding, 'CONSTANT')
        conv4 = tf.nn.conv2d(pad3, w_conv4, strides=[1, 1, 1, 1], padding='SAME', name="conv4") + b_conv4
    
    with tf.name_scope("relu4"):
        relu4 = tf.nn.relu(conv4, name="relu4")

    with tf.name_scope("conv5"): 
        w_conv5 = weight_variable_conv2d([3, 3, 192, 128], "w_conv5")
        b_conv5 = bias_variable([128], "b_conv5")
        padding = [[0,0], [1,1], [1,1], [0,0]]
        pad4 = tf.pad(relu4, padding, 'CONSTANT')
        conv5 = tf.nn.conv2d(pad4, w_conv5, strides=[1, 1, 1, 1], padding='SAME', name="conv5") + b_conv5

    with tf.name_scope("relu5"):
        relu5 = tf.nn.relu(conv5, name="relu5")

    with tf.name_scope("pool5"):
        pool5 = tf.nn.max_pool(relu5, ksize=[1, 6, 6, 1], strides=[1, 4, 4, 1], padding='SAME', name="pool5")

    with tf.name_scope("flat"):
        pool5_flat = tf.reshape(pool5, [-1, 6*16*128])   # edit values here

    with tf.name_scope("fc6"):
        w_fc6 = weight_variable([6*16*128, 4096], "w_fc6")
        b_fc6 = bias_variable([4096], "b_fc6")
        fc6 = tf.matmul(pool5_flat, w_fc6) + b_fc6

    with tf.name_scope("relu6"):   
        relu6 = tf.nn.relu(fc6, name="relu6")

    #with tf.name_scope("fc7"):
    #    w_fc7 = weight_variable([4096, 4096], "w_fc7")
    #    b_fc7 = bias_variable([4096], "b_fc7")
    #    fc7 = tf.matmul(relu6, w_fc7) + b_fc7

    #with tf.name_scope("relu7"):   
    #    relu7 = tf.nn.relu(fc7, name="relu7")  
    
    return relu6

def fc_network_model(siamese_output1, siamese_output2, drop7_prob):
    with tf.name_scope("concat_out"):
        concat_out = tf.concat(1, [siamese_output1, siamese_output2], name="concat_out")

    # concat_out has 8192 activations

    with tf.name_scope("fc8"):
        W_fc8 = weight_variable([8192, 8192], "W_fc8")
        b_fc8 = bias_variable([8192], "b_fc8")
        fc8 = tf.matmul(concat_out, W_fc8) + b_fc8

    with tf.name_scope("relu8"):   
        relu8 = tf.nn.relu(fc8)

    with tf.name_scope("drop7"):
        drop7 = tf.nn.dropout(relu8, drop7_prob)

    with tf.name_scope("fc9"):
        W_fc9 = weight_variable([8192, 256], "W_fc9")
        b_fc9 = bias_variable([256], "b_fc9")
        fc9 = tf.matmul(drop7, W_fc9) + b_fc9

    with tf.name_scope("relu9"):   
        relu9 = tf.nn.relu(fc9)

    with tf.name_scope("fc10"):
        W_fc10 = weight_variable([256, 2], "W_fc10")
        b_fc10 = bias_variable([2], "b_fc10")
        fc10 = tf.matmul(relu9, W_fc10) + b_fc10

    # with tf.name_scope("relu10"):   
    #     relu10 = tf.nn.relu(fc10)    

    # with tf.name_scope("fc11"):
    #     W_fc11 = weight_variable([256, 2], "W_fc11")
    #     b_fc11 = bias_variable([2], "b_fc11")
    #     fc11 = tf.matmul(relu10, W_fc11) + b_fc11

    return fc10
    
def train(trainX, trainY, validationX, validationY):
    train_data, train_labels = trainX, trainY
    eval_data, eval_labels = validationX, validationY
    
    tf.reset_default_graph()
    sess = tf.Session()

    with tf.name_scope("images"):
        x_image_t = tf.placeholder(tf.float64, [None, 256, 844, 1], name="image_t")
        x_image_t_1 = tf.placeholder(tf.float64, [None, 256, 844, 1], name="image_t_1")

    with tf.name_scope("labels"):    
        target = tf.placeholder(tf.float32, [None, 2], name="target")
    
    with tf.variable_scope("siamese_network") as scope:
        siam_output1 = siamese_network_model(x_image_t)
        scope.reuse_variables()
        siam_output2 = siamese_network_model(x_image_t_1)

    with tf.name_scope("fc_network"):
        with tf.name_scope("drop7"):  # drop param must be part of network model
            drop7_prob = tf.placeholder(tf.float32)

        final_output = fc_network_model(siam_output1, siam_output2, drop7_prob)

    with tf.name_scope("loss_function"):
    #SEE REDUCE_MEAN vs REDUCE_SUM
        l2_loss =  tf.reduce_mean(tf.square(tf.sub(final_output, target)))
        #l2_loss_avg = tf.reduce_mean(tf.square(tf.sub(final_output, target)))
    
    with tf.name_scope("optimizer"):
        train_step = tf.train.AdamOptimizer(0.00001).minimize(l2_loss)
    
    # store important operations for deploying
    tf.add_to_collection('train_vo', x_image_t)
    tf.add_to_collection('train_vo', x_image_t_1)
    tf.add_to_collection('train_vo', drop7_prob)
    tf.add_to_collection('train_vo', final_output)

    # collect summary of these operations
    train_summ = tf.scalar_summary("training_loss", l2_loss)
    #eval_summ = tf.scalar_summary("validation_loss", l2_loss)
    #summary_op = tf.merge_all_summaries()

    if os.path.exists("weights/train_vo"):
    	saver = tf.train.import_meta_graph('weights/train_vo.meta')
    	saver.restore(sess, "weights/train_vo")
    	print "Restoring saved weights....."
    else:
    	sess.run(tf.initialize_all_variables())
        writer = tf.train.SummaryWriter(log_path, graph=tf.get_default_graph()) # for tensorboard
        saver = tf.train.Saver() # saving mechanism for graph and variables
        print "Initializing variables, no saved weights found ...."
    
    minibatch = 4
    #index = range(len(train_labels))

    for epoch in range(100):
        train_data, train_labels = shuffle(train_data, train_labels)
        #eval_data, eval_labels = shuffle(eval_data, eval_labels)
        for k in xrange(0, len(train_labels), minibatch):   
            batch_input, batch_output = train_data[k : k + minibatch] , train_labels[k : k + minibatch] 
            batch_input = np.asarray([[i[0], i[1]] for i in batch_input])
            batch_input = normalize(batch_input)
            batch_input = batch_input[:, :, :, :, np.newaxis]
            batch_output = np.asarray([[i[0], i[1]] for i in batch_output])

            _, train_summary = sess.run([train_step, train_summ], feed_dict = {x_image_t: batch_input[:, 0], x_image_t_1: batch_input[:, 1] ,target: batch_output, drop7_prob: 0.50 })

            itr = epoch*len(train_labels)/minibatch + k/minibatch
            if (itr % 100 == 0):
                writer.add_summary(train_summary, itr) # append summary     
                print "interations: ", itr 

            if (itr % 1000 == 0):
                l2_loss_avg = 0
                for mark in xrange(0, len(eval_labels), minibatch):   
                    batch_input, batch_output = eval_data[mark : mark + minibatch] , eval_labels[mark : mark + minibatch] 
                    batch_input = np.asarray([[i[0], i[1]] for i in batch_input])
                    batch_input = normalize(batch_input)
                    batch_input = batch_input[:, :, :, :, np.newaxis]
                    batch_output = np.asarray([[i[0], i[1]] for i in batch_output])
                    #_, eval_summary = sess.run([l2_loss, eval_summ], feed_dict = {x_image_t: batch_input[:,0], x_image_t_1: batch_input[:,1], target: batch_output, drop7_prob: 1.0 })    
                    l2_loss_avg = l2_loss_avg + sess.run(l2_loss, feed_dict = {x_image_t: batch_input[:, 0], x_image_t_1: batch_input[:,1], target: batch_output, drop7_prob: 1.0 })
                
                l2_loss_avg = l2_loss_avg/((int)(len(eval_labels))/minibatch)
                eval_summ = sess.run(tf.scalar_summary("validation_loss", l2_loss_avg))
                writer.add_summary(eval_summ, itr)

        print "epoch: ", epoch
        if (epoch % 10 == 0): 
            saver.save(sess, "weights/train_vo")
            print "Checkpoint saved"
    sess.close()

def test():

	# reset graph if any graph was produced before this serving.
	tf.reset_default_graph()
	sess = tf.Session()

	saver = tf.train.import_meta_graph('weights/train_vo.meta')
	saver.restore(sess, "weights/train_vo")

	# recapitulate operations 
	x_image_t = tf.get_collection('train_vo')[0]
	x_image_t_1 = tf.get_collection('train_vo')[1]
	drop7_prob = tf.get_collection('train_vo')[2]
	final_output = tf.get_collection('train_vo')[3]

	test_files = ["05.npy", "09.npy"]
	for file in test_files:
		print "testing: ", file
		test_load = np.load(data_path + "/" + file)
		test_data = test_load[:, 0:2]
		test_labels = test_load[:, 2]

		f = open(result_path + "/test_" + file[:file.index(".")] + ".csv", "wb")
		csvwriter = csv.writer(f, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)

		for i in range(len(test_labels)):

			data_input = np.asarray([test_data[i][0], test_data[i][1]])
			data_input = normalize(data_input)

			image_t = data_input[0]
			image_t = image_t[np.newaxis, :]
			image_t_1 = data_input[1]
			image_t_1 = image_t_1[np.newaxis, :]

			regr_output = sess.run(final_output, feed_dict = {x_image_t: image_t, x_image_t_1: image_t_1, drop7_prob: 1.0})
			csvwriter.writerow([regr_output[0][0], regr_output[0][1]])
			print regr_output
		
		f.close()	

	sess.close()
    #return regr_output

if __name__ == "__main__":
	
	train_full = np.load(data_path+'/training_set.npy')
	train_data = train_full[:, 0:2]
	train_labels = train_full[:, 2]

	eval_full = np.load(data_path+'/test_set.npy')
	eval_data = eval_full[:, 0:2]
	eval_labels = eval_full[:, 2]
	
	train(train_data, train_labels, eval_data, eval_labels)

	# test()

	# test_files = ["05.npy", "02.npy", "09.npy"]
	# for file in test_files:
	# 	print "testing: ", file
	# 	test_load = np.load(data_path + "/" + file)
	# 	test_data = test_load[:, 0:2]
	# 	test_labels = test_load[:, 2]

	# 	f = open(result_path + "/test_" + file[:file.index(".")] + ".csv", "wb")
	# 	csvwriter = csv.writer(f, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
		
	# 	for i in range(len(test_labels)):
	# 		regr_output = test(test_data[i])
	# 		csvwriter.writerow([regr_output[0][0], regr_output[0][1]])
	# 		print regr_output

	


	
