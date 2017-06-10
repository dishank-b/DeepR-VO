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
    with open("./dz_x_angle/"+seq_no+"_zxangle.txt", 'w') as f:
        for i in range(len(poses_array)):
            if i == 0:
                line = " ".join(str(x) for x in [0, 0, 0])
                f.write(line+"\n")
            else:
                R_cam = np.dot(np.linalg.inv(poses_array[i-1][:,:3]),poses_array[i][:,:3])
                T_cam = np.dot(np.linalg.inv(poses_array[i-1][:,:3]),poses_array[i][:,3:4]-poses_array[i-1][:,3:4])
                dangle_rad = np.arctan(-R_cam[2][0]/(R_cam[2][1]**2+R_cam[2][2]**2)**0.5)
                dangle_deg = dangle_rad*180/np.pi
                dz = T_cam[2][0]
                dx = T_cam[0][0]
                line = " ".join(str(x) for x in [dz, dx, dangle_deg])
                f.write(line+"\n")
        f.close()        
        print "Process Complete"
        return


poses_path = "/mnt/B4223FDF223FA570/Data_sets/Kitti_dataset/poses/"
seq_no = sys.argv[1]
file_name = poses_path+seq_no+".txt"

poses_array = load_poses(file_name)
cam_poses = conv_poses(poses_array)
