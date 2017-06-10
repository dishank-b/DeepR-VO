import tensorflow as tf
import numpy as np
import cv2
# import inspect
def normalize(data):
    normalized_data = (data.astype(np.float32))*255 + 128
    return normalized_data

new_image = np.zeros(shape=[11, 11])

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('train_vo.meta')
	saver.restore(sess, "train_vo")
	with tf.name_scope("conv1"):
		weight_tensor = tf.get_variable(name='w_conv1', shape=[11, 11, 1, 48])
		init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
		sess.run(init_op)
		weight_matrix = sess.run(weight_tensor)
		print weight_matrix
		# print weight_matrix[:][:][:][0].shape
		for i in range(48):
			for j in range(11):
				for k in range(11):
					new_image[j][k] = weight_matrix[j][k][0][i]
			image = normalize(new_image)
			cv2.imwrite("./filters/filter_"+str(i)+".jpg", image)
