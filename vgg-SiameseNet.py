import tensorflow as tf
import numpy as np
import glob
import os
import sys
import csv
import cv2

def shuffle(train_data, train_labels):

    permutation = np.random.permutation(train_labels.shape[0])
    shuffled_dataset = train_data[permutation]
    shuffled_labels = train_labels[permutation]
    return shuffled_dataset, shuffled_labels


class vgg16:
    def __init__(self, drop_prob):
        # self.imgs = imgs
        self.layer_list = []
        self.drop_prob = drop_prob
        # self.convlayers1()
        # self.convlayers2()
        # self.fc_layers()
        # if weights is not None and sess is not None:
        #     self.load_weights(weights, sess)


    def convnet_1(self, image_t):
        # conv1_1
        with tf.name_scope("cnn_1") as sc:
            with tf.name_scope('conv1_1') as scope:
                with tf.variable_scope("conv1_1") as var_scope:
                    self.layer_list.append("conv1_1")
                    kernel = tf.get_variable(shape=[3, 3, 3, 64], name='weights', trainable=False, initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    biases = tf.get_variable(shape=[64], trainable=False, name='biases',initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(image_t, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv1_1 = tf.nn.relu(out, name=scope)

                # pool1
                pool1 = tf.nn.max_pool(conv1_1,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name='pool1')

            # conv2_1
            with tf.name_scope('conv2_1') as scope:
                with tf.variable_scope("conv2_1") as var_scope:
                    self.layer_list.append("conv2_1")
                    kernel = tf.get_variable(shape=[3, 3, 64, 128],trainable=False, name='weights',initializer=tf.contrib.layers.xavier_initializer_conv2d() )
                    biases = tf.get_variable(shape=[128],trainable=False, name='biases', initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv2_1 = tf.nn.relu(out, name=scope)

                # pool2
                pool2 = tf.nn.max_pool(conv2_1,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name='pool2')

            # conv3_1
            with tf.name_scope('conv3_1') as scope:
                with tf.variable_scope("conv3_1") as var_scope:
                    self.layer_list.append("conv3_1")
                    kernel = tf.get_variable(shape=[3, 3, 128, 256], trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    biases = tf.get_variable(shape=[256], trainable=True, name='biases', initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv3_1 = tf.nn.relu(out, name=scope)

                # pool3
                pool3 = tf.nn.max_pool(conv3_1,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name='pool3')

            # conv4_1
            with tf.name_scope('conv4_1') as scope:
                with tf.variable_scope("conv4_1") as var_scope:
                    self.layer_list.append("conv4_1")
                    kernel = tf.get_variable(shape=[3, 3, 256, 512],trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    biases = tf.get_variable(shape=[512],trainable=True, name='biases', initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv4_1 = tf.nn.relu(out, name=scope)

                # pool4
                pool4 = tf.nn.max_pool(conv4_1,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name='pool4')

            # conv5_1
            with tf.name_scope('conv5_1') as scope:
                with tf.variable_scope("conv5_1") as var_scope:
                    self.layer_list.append("conv5_1")
                    kernel = tf.get_variable(shape=[3, 3, 512, 512],trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    biases = tf.get_variable(shape=[512], trainable=True, name='biases', initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv5_1 = tf.nn.relu(out, name=scope)

                # pool5
                pool5 = tf.nn.max_pool(conv5_1,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name='pool5')

            with tf.name_scope('conv6_1') as scope:
                with tf.variable_scope("conv6_1") as var_scope:
                    self.layer_list.append("conv6_1")
                    kernel = tf.get_variable(shape=[1, 1, 512, 64],trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    biases = tf.get_variable(shape=[64], trainable=True, name='biases',initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                # conv6_1 = tf.nn.relu(out, name=scope)

                # # pool5
                # pool6 = tf.nn.max_pool(conv6_1,
                #                        ksize=[1, 2, 2, 1],
                #                        strides=[1, 2, 2, 1],
                #                        padding='SAME',
                #                        name='pool6')

        return out 

    def convnet_2(self, image_t_next):
        # conv1_1
        with tf.name_scope("cnn_2") as sc:
            
            with tf.name_scope('conv1_1') as scope:
                with tf.variable_scope("conv1_1", reuse=True) as var_scope:
                    self.layer_list.append("conv1_1")
                    kernel = tf.get_variable(shape=[3, 3, 3, 64], name='weights', trainable=True, initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    biases = tf.get_variable(shape=[64], trainable=True, name='biases',initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(image_t_next, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv1_1 = tf.nn.relu(out, name=scope)

                # pool1
                pool1 = tf.nn.max_pool(conv1_1,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name='pool1')

            # conv2_1
            with tf.name_scope('conv2_1') as scope:
                with tf.variable_scope("conv2_1", reuse=True) as var_scope:
                    self.layer_list.append("conv2_1")
                    kernel = tf.get_variable(shape=[3, 3, 64, 128],trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer_conv2d() )
                    biases = tf.get_variable(shape=[128],trainable=True, name='biases',initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv2_1 = tf.nn.relu(out, name=scope)

                # pool2
                pool2 = tf.nn.max_pool(conv2_1,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name='pool2')

            # conv3_1
            with tf.name_scope('conv3_1') as scope:
                with tf.variable_scope("conv3_1", reuse=True) as var_scope:
                    self.layer_list.append("conv3_1")
                    kernel = tf.get_variable(shape=[3, 3, 128, 256], trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    biases = tf.get_variable(shape=[256], trainable=True, name='biases',initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv3_1 = tf.nn.relu(out, name=scope)

                # pool3
                pool3 = tf.nn.max_pool(conv3_1,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name='pool3')

            # conv4_1
            with tf.name_scope('conv4_1') as scope:
                with tf.variable_scope("conv4_1", reuse=True) as var_scope:
                    self.layer_list.append("conv4_1")
                    kernel = tf.get_variable(shape=[3, 3, 256, 512],trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    biases = tf.get_variable(shape=[512],trainable=True, name='biases',initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv4_1 = tf.nn.relu(out, name=scope)

                # pool4
                pool4 = tf.nn.max_pool(conv4_1,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name='pool4')

            # conv5_1
            with tf.name_scope('conv5_1') as scope:
                with tf.variable_scope("conv5_1", reuse=True) as var_scope:
                    self.layer_list.append("conv5_1")
                    kernel = tf.get_variable(shape=[3, 3, 512, 512],trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    biases = tf.get_variable(shape=[512], trainable=True, name='biases',initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv5_1 = tf.nn.relu(out, name=scope)

                # pool5
                pool5 = tf.nn.max_pool(conv5_1,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name='pool5')

            with tf.name_scope('conv6_1') as scope:
                with tf.variable_scope("conv6_1", reuse=True) as var_scope:
                    self.layer_list.append("conv6_1")
                    kernel = tf.get_variable(shape=[1, 1, 512, 64],trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    biases = tf.get_variable(shape=[64], trainable=True, name='biases',initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                # conv6_1 = tf.nn.relu(out, name=scope)

                # pool6
                # pool6 = tf.nn.max_pool(conv6_1,
                #                        ksize=[1, 2, 2, 1],
                #                        strides=[1, 2, 2, 1],
                #                        padding='SAME',
                #                        name='pool6')

        return out    # Add Dropouts

    def fc_layers(self, feature_1, feature_2):

        #Concatanation
        with tf.name_scope('concat') as scope:
            concat_out = tf.concat([feature_1, feature_2],3, name="concat")
            print concat_out.get_shape()
            concat_drop = tf.nn.dropout(concat_out, self.drop_prob)

        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(concat_drop.get_shape()[1:]))
            with tf.variable_scope("fc6") as var_scope:
                fc1w = tf.get_variable(shape=[shape, 4096],trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer())
                fc1b = tf.get_variable(shape=[4096],trainable=True, name='biases',initializer=tf.constant_initializer(0.0))
            concat_flat = tf.reshape(concat_drop, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(concat_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(fc1l)

        # fc2
        with tf.name_scope('fc2') as scope:
            with tf.variable_scope('fc7') as var_scope:
                fc2w = tf.get_variable(shape=[4096, 512],trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer())
                fc2b = tf.get_variable(shape=[512], trainable=True, name='biases',initializer=tf.constant_initializer(0.0))
            fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
            # fc2 = tf.nn.relu(fc2l)

        # fc3
        with tf.name_scope('fc3') as scope:
            with tf.variable_scope('fc8') as var_scope:
                fc3w = tf.get_variable(shape=[512, 3],trainable=True, name='weights',initializer=tf.contrib.layers.xavier_initializer())
                fc3b = tf.get_variable(shape=[3],trainable=True, name='biases', initializer=tf.constant_initializer(0.0))
            fc3 = tf.nn.bias_add(tf.matmul(fc2l, fc3w), fc3b)

        return fc3

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        # print weights
        # keys = sorted(weights.keys())
        for op_name in weights:
            if 'conv' in op_name:
                if op_name[:7] in self.layer_list:
                    with tf.variable_scope(op_name[:7], reuse=True) as scope:
                        if op_name[-1]=='W':
                            var = tf.get_variable('weights', weights[op_name].shape)
                            sess.run(var.assign(weights[op_name]))
                        else:
                            var = tf.get_variable('biases', weights[op_name].shape)
                            sess.run(var.assign(weights[op_name]))
                

        # for i, k in enumerate(keys):
        #     print i, k, np.shape(weights[k])
        #     sess.run(self.parameters[i].assign(weights[k]))

    def train_model(self,data_seq, val_input, val_output, leanring_rate, no_epochs, batch_size, log_info,use_pretrain=False):
        # with tf.device("/gpu:1"):
        with tf.name_scope("Input") as scope:
            x_image = tf.placeholder(tf.float32, [None, 188, 620, 3])
            x_next_image = tf.placeholder(tf.float32, [None, 188, 620, 3])
            y = tf.placeholder(tf.float32, [None, 3])

        with tf.name_scope("Network") as scope:
            feature_1 = self.convnet_1(x_image)
            feature_2 = self.convnet_2(x_next_image)
            final_output = self.fc_layers(feature_1, feature_2)
        
        with tf.name_scope("Training") as scope:
            data_loss = tf.reduce_mean(tf.square(tf.subtract(final_output,y)))
            loss = data_loss
            train_op = tf.train.AdamOptimizer(leanring_rate).minimize(loss)
        
        with tf.name_scope("Loss_Monitoring") as scope:
            train_sum_op = tf.summary.scalar("Training Loss", loss)
            merged_sum = tf.summary.merge_all()
            # val_sum_op = tf.summary.scalar("Vaildation", loss)

        if len(glob.glob("./logs/"+log_info+"/model*")) != 0:
            tf.reset_default_graph()
            saver = tf.train.import_meta_graph("./logs/"+log_info+"/model.meta")
            saver.restore(sess,"./logs/"+log_info+"/model")
            print "Continuing Training On Previously Saved Weights."
        else:   
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            if use_pretrain: 
                self.load_weights('./vgg-siamese/vgg16_weights.npz', sess)
                print " Training Using Pre-trained VGG Weights."
            else:
                print "Training Started From Beginning. No Pre-Trained Weights Used."
        
        writer = tf.summary.FileWriter("./logs/"+log_info)
        writer.add_graph(sess.graph)

        total_parameters = 0
        for variable in tf.trainable_variables():
            name = variable.name
            shape = variable.get_shape()
            print name, shape
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print total_parameters

        file = open("./logs/"+log_info+'/loss_record.csv', 'a')
        file_writer = csv.writer(file, delimiter=',')
        data_path = "./training_testing_data/620x188_images/"
        # print data_seq
        itr=0
        for epoch in range(no_epochs):
        	for seq in data_seq:
        		# print "Loading Data", seq
        		train_full = np.load(data_path+seq+'.npy')
        		print seq, "Loaded.", train_full.shape[0], "Training Examples."
        		input_data = train_full[:, 0:2]
        		output_data = train_full[:, 2]
        		print input_data.shape, output_data.shape
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
        			train_loss, _ , summary = sess.run([loss, train_op, merged_sum],{x_image:batch_input[:,0], x_next_image:batch_input[:,1], y:batch_output})
        			# itr = epoch*len(input_data)/batch_size + k/batch_size
        			itr = itr+1
        			writer.add_summary(summary)
        			if (itr % 100 == 0):
						writer.add_summary(summary, itr) # append summary
						print "Epoch:-", epoch,"Sequence",seq, "Iteration:-", k/batch_size
						val_loss_tot = 0
						for j in xrange(0, len(val_input)-batch_size, batch_size):
							val_batch_input = val_input[j:j+batch_size]
							val_batch_input = np.asarray([[i[0], i[1]] for i in val_batch_input])
							val_batch_output = val_output[j:j+batch_size]
							val_batch_output = np.asarray([[i[0], i[1], i[2]] for i in val_batch_output])
							val_loss_tot = val_loss_tot + sess.run(loss, {x_image:val_batch_input[:,0], x_next_image:val_batch_input[:,1], y:val_batch_output})

						val_loss_avg = val_loss_tot/int(len(val_input)/batch_size)

						write_string = ['Epoch:- '+str(epoch), "Validation_Loss:- "+str(val_loss_avg), "Training_Loss:- "+str(train_loss)]
						file_writer.writerow(write_string)
						print "Validation Loss:", val_loss_avg
						print "Training Loss:", train_loss
			if (epoch % 5 == 0):
				saver.save(sess, "./logs/"+log_info+"/model")
				print "Checkpoint saved"

if __name__ == '__main__':
    #Generating Data here.
    log_info = sys.argv[1]
    print log_info
    data_seq = ["04","01","02", "00","05","06","07","08","09","10"]
    data_path = "./training_testing_data/620x188_images/"
    eval_full = np.load(data_path+'03.npy')
    val_input = eval_full[:, 0:2]
    val_output = eval_full[:, 2]
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        vgg = vgg16(drop_prob=0.5)
        vgg.train_model(data_seq,val_input, val_output, leanring_rate = 0.00001, no_epochs= 100, batch_size=16,log_info=log_info, use_pretrain=True)

