# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import cv2
import numpy as np
import os
import glob
import sys

# def resize_image(image):
#     """Padding the image to get resize image. 227*277 or 620*188"""
#     height = image[0]
#     width = image[1]
#     float aspet_ratio = height/width
#     new_width = 256
#     int new_height = new_width * aspet_ratio
#     rsz_image = cv2.resize(image, (new_width, new_height), INTER_NEAREST)
#     int padding = (256 - new_height)/2
#     padded_image = cv2.copyMakeBorder(rsz_image, padding, padding, 0, 0, [0,0,0])
#     print rsz_image.shape(), padding
#     return padded_image

def load_poses(file_name):
    """Load ground truth poses from file."""
    pose_file = os.path.join(poses_path+ file_name)
    poses = []
    # Read and parse the poses
    with open(pose_file, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=float, sep=' ')
            T = T.reshape(3, 4)
            poses.append(T)
            # print T
    poses_array = np.array(poses)
    return poses_array

def load_images(sequence_no):
	# """Load left stereo images from file. Returns a array of all images in a given sequece."""
    image_list = []
    images_names = sorted(glob.glob(images_path))
    for i in images_names:
    	image = cv2.imread(i)
        resize_image  = cv2.resize(image, dsize = (414, 126), interpolation = cv2.INTER_NEAREST)
    	image_list.append(resize_image)
    images_array = np.array(image_list)
    return images_array

######Conversion from world frame to camera frame#########
#using Xᶜᵃᵐₜ₋₁ = (R⁻¹ₜ₋₁*Rₜ)*Xₜ + R⁻¹ₜ₋₁*(Tₜ - Tₜ₋₁)
#Here R and T matrix for t-1 and t frame are given in poses file
#R⁻¹ₜ₋₁*(Tₜ - Tₜ₋₁) => gives transation from t-1 frame to t frame wrt coordinate system of frame t-1
#R⁻¹ₜ₋₁*Rₜ => +ve angle along y-axis denotes vehicle is turning right while moving from frame t-1 to frame t.

def gen_data(poses_array, image_array):
	"""Generate Training Data"""
	# if name=="train":
	data_list = []
	for j in xrange(1,3):
		for i in range(len(poses_array)-j):
			R_cam = np.dot(np.linalg.inv(poses_array[i][:,:3]),poses_array[i+j][:,:3])
			T_cam = np.dot(np.linalg.inv(poses_array[i][:,:3]),poses_array[i+j][:,3:4]-poses_array[i][:,3:4])
			dangle_rad = np.arctan(-R_cam[2][0]/(R_cam[2][1]**2+R_cam[2][2]**2)**0.5)
			dangle_deg = dangle_rad*180/np.pi
			dz = T_cam[2][0]
			dx = T_cam[0][0]
			dchange = np.array([dz, dx, dangle_deg])
			traning_example = np.array([image_array[i], image_array[i+j], dchange])
			data_list.append(traning_example)
	data_array = np.array(data_list)
	return data_array

poses_path = "./raw_data/poses/"
saving_path = "./training_testing_data/414x126_images/"

for sequence_no in ["00","01","02","03","04","05","06","07","08","09","10"]:
	poses = load_poses(str(sequence_no)+".txt")
	images_path = "./raw_data/sequences/"+str(sequence_no)+"/image_2/*.png"
	image_array = load_images(str(sequence_no))
	training_array= gen_data(poses, image_array)
	np.save(saving_path+sequence_no+".npy", training_array)
	print "Trainnig data stored", sequence_no
	


# for sequence_no in ["03", "04"]:
# 	poses = load_poses(str(sequence_no)+".txt")
# 	images_path = "./raw_data/sequences/"+str(sequence_no)+"/image_2/*.png"
# 	image_array = load_images(str(sequence_no))
# 	val_list = gen_data(poses, image_array, val_list, name="validation")
	
# print "Val length", len(val_list)

# for sequence_no in ["03", "04"]:
# 	poses = load_poses(str(sequence_no)+".txt")
# 	images_path = "./raw_data/sequences/"+str(sequence_no)+"/image_2/*.png"
# 	image_array = load_images(str(sequence_no))
# 	test_list = gen_data(poses, image_array, test_list, name="testing")
# 	print "Test length", len(test_list)
	

# val_array = np.array(val_list)
# test_array = np.array(test_list)


# np.save(saving_path+"val_data.npy", val_array)
# print "Trainnig data stored"
# np.save(saving_path+"test_data.npy", val_array)
# print "Trainnig data stored"

