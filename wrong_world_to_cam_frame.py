import cv2
import numpy as np
import os
import glob
import sys


def load_poses(file_name):
        """Load ground truth poses from file."""
        pose_file = file_name
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

def conv_poses(poses_array):
    """Convert transpose matrix from ground frame to camera frame"""
    cam_poses_array = []
    for i in range(len(poses_array)):
        if i == 0:
            R_cam = np.eye(3)
            T_cam = np.array([[0],[0],[0]])
            Mat = np.concatenate((R_cam,T_cam), axis=1)
            cam_poses_array.append(Mat)
        else:
            R_cam = np.dot(poses_array[i][:,:3],np.linalg.inv(poses_array[i-1][:,:3]))
            T_cam = poses_array[i][:,3:4] - np.dot(R_cam, poses_array[i-1][:,3:4])
            Mat = np.concatenate((R_cam,T_cam), axis=1)
            cam_poses_array.append(Mat) 
    return cam_poses_array

def write_poses_txt(cam_poses_array, seq_no):
    """This file writes camera frame matrix to txt file."""
    with open("./wrong_cam_poses/"+seq_no+"_cam.txt", 'w') as f:
        for i in range(len(cam_poses_array)):
            mat = cam_poses_array[i]
            mat = mat.reshape(12)
            print mat.dtype
            mat = mat.tolist()
            line = " ".join(str(x) for x in mat)
            f.write(line+"\n")
        f.close()
        print "Process Complete"
        return

poses_path = "/mnt/B4223FDF223FA570/Data_sets/Kitti_dataset/poses/"
seq_no = sys.argv[1]
file_name = poses_path+seq_no+".txt"

np.set_printoptions(suppress=True)
poses_array = load_poses(file_name)
cam_poses = conv_poses(poses_array)
write_poses_txt(cam_poses, seq_no)