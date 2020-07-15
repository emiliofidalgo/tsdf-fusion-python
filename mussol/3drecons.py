#!/usr/bin/python
"""
BSD 2-Clause License

Copyright (c) 2020, Emilio Garcia-Fidalgo
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
import cv2
import datetime
import numpy as np
import os
import shutil
import sys
import time


class Observation:

    def __init__(self, timestamp, rgb, depth, pose):
        self.timestamp = timestamp
        self.rgb_filename = rgb
        self.depth_filename = depth
        self.pose = pose
    
    def __str__(self):
        val = '---\n'
        val += 'Timestamp: %i\n' % self.timestamp
        val += 'RBG file: %s\n' % self.rgb_filename
        val += 'Depth file: %s\n' % self.depth_filename
        val += 'Pose:\n %s\n' % np.array_str(self.pose)
        return val


class TSDFBuilder:

    def __init__(self, args):

        print('Initializing TSDF Builder ...')

        # Directories
        self._data_dir = args.data_dir
        self._csv_dir = os.path.join(self._data_dir, 'csv')
        self._imgs_dir = os.path.join(self._data_dir, 'images')        

        # Remainig params
        self._voxel_size = args.voxel_size

        # Loading intrinsics
        caminfo_filename = os.path.join(self._imgs_dir, args.caminfo_filename)
        self._cam_intr = self._load_intrinsics(caminfo_filename)

        # Loading positions
        pose_filename = os.path.join(self._csv_dir, args.pose_filename)
        self._pose_timestamps, self._poses = self._load_poses(pose_filename)

        # Preparing frames
        self._frames = self._prepare_frames()        
        print('%i frames found' % len(self._frames))
    
    def _load_intrinsics(self, caminfo_filename):
        data_cam_info = np.loadtxt(caminfo_filename, delimiter=',', skiprows=1, usecols=(12,13,14,15,16,17,18,19,20))
        
        return data_cam_info.reshape((3, 3))
    
    def _load_poses(self, pose_filename):
        # Reading pose file
        pose_timestamps = []
        poses = {}
        with open(pose_filename) as f:
            _ = f.readline() # Skip the first line (header)
            for line in f:
                parts = line.split(',')
                timestamp = int(parts[0])

                # Adding current timestamp to the list
                pose_timestamps.append(timestamp)

                # Adding pose information to the poses dict
                poses[timestamp] = parts

        return pose_timestamps, poses
    
    def _find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        
        return array[idx]
    
    def _get_pose(self, timestamp):
        # Finding the closest timestamp
        nearest_timestamp = self._find_nearest(self._pose_timestamps, timestamp)

        # Get pose info
        pose_data = self._poses[nearest_timestamp]

        # Filling the pose
        pose = np.eye(4)

        # Filling translation
        pose[0,3] = float(pose_data[6])
        pose[1,3] = float(pose_data[7])
        pose[2,3] = float(pose_data[8])

        # Filling rotation
        qw = float(pose_data[12])
        qx = float(pose_data[9])
        qy = float(pose_data[10])
        qz = float(pose_data[11])

        pose[0,0] = 1 - 2*qy**2 - 2*qz**2
        pose[0,1] = 2*qx*qy - 2*qz*qw
        pose[0,2] = 2*qx*qz + 2*qy*qw
        pose[1,0] = 2*qx*qy + 2*qz*qw
        pose[1,1] = 1 - 2*qx**2 - 2*qz**2
        pose[1,2] = 2*qy*qz - 2*qx*qw
        pose[2,0] = 2*qx*qz - 2*qy*qw
        pose[2,1] = 2*qy*qz + 2*qx*qw
        pose[2,2] =	1 - 2*qx**2 - 2*qy**2

        return pose
    
    def _prepare_frames(self):

        observations = []

        # Reading associate.txt file
        associate_file = os.path.join(self._imgs_dir, 'associate.txt')
        with open(associate_file) as f:
            for line in f:
                parts = line.split()
                
                # Getting data
                rgb_filename = os.path.join(self._imgs_dir, parts[1])
                depth_filename = os.path.join(self._imgs_dir, parts[3])
                timestamp = int(os.path.splitext(os.path.basename(rgb_filename))[0])
                pose = self._get_pose(timestamp)

                # Creating a new observation
                obs = Observation(timestamp, rgb_filename, depth_filename, pose)
                observations.append(obs)                

        return observations

if __name__ == "__main__":
    # Reading parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="directory where the files are stored")
    parser.add_argument("-vs", "--voxel_size", default=0.02,
                        help="voxel size in meters")
    parser.add_argument("-pose", "--pose_filename", default='_ekf_local_odom.csv',
                        help="pose filename where MAV absolute positions are saved")
    parser.add_argument("-cam", "--caminfo_filename",default='_camera_color_camera_info.csv', help="camera info filename where calibration info is saved")
    args = parser.parse_args()

    # Assessing if the data directory exists
    if (not os.path.exists(args.data_dir)):
        print("Directory does not exist!")
        sys.exit(0)
    
    builder = TSDFBuilder(args)
