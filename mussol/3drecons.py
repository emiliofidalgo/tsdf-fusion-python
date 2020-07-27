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
import fusion_nonum as fusion
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import os
import transformations as trans
from pyquaternion import Quaternion
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
    
    def load_RGB(self):
        return cv2.cvtColor(cv2.imread(self.rgb_filename), cv2.COLOR_BGR2RGB)
    
    def load_D(self):
        img = cv2.imread(self.depth_filename,-1).astype(float)
        img /= 5000.
        return img

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

        dpose = -1

        # Filling translation
        # pose[0,3] = float(pose_data[6 + dpose])
        # pose[1,3] = float(pose_data[7 + dpose])
        # pose[2,3] = float(pose_data[8 + dpose])

        tx = float(pose_data[6 + dpose])
        ty = float(pose_data[7 + dpose])
        tz = float(pose_data[8 + dpose])

        pose[0,3] = ty # X
        pose[1,3] = tz # Y 
        pose[2,3] = tx # Z

        # Filling rotation
        qw = float(pose_data[12 + dpose])
        qx = float(pose_data[9 + dpose])
        qy = float(pose_data[10 + dpose])
        qz = float(pose_data[11 + dpose])

        #quat = Quaternion(qw, qx, qy, qz)

        euler    = trans.euler_from_quaternion([qx, qy, qz, qw], 'rzyx')
        new_quat = trans.quaternion_from_euler(euler[1], euler[2], euler[0], 'rzyx')
        quat = Quaternion(new_quat[3], new_quat[0], new_quat[1], new_quat[2])

        # pose[0,0] = 1 - 2*qy**2 - 2*qz**2
        # pose[0,1] = 2*qx*qy - 2*qz*qw
        # pose[0,2] = 2*qx*qz + 2*qy*qw
        # pose[1,0] = 2*qx*qy + 2*qz*qw
        # pose[1,1] = 1 - 2*qx**2 - 2*qz**2
        # pose[1,2] = 2*qy*qz - 2*qx*qw
        # pose[2,0] = 2*qx*qz - 2*qy*qw
        # pose[2,1] = 2*qy*qz + 2*qx*qw
        # pose[2,2] =	1 - 2*qx**2 - 2*qy**2

        pose[:3,:3] = quat.rotation_matrix

        static_trans = np.eye(4)
        # static_trans[:3,:3] = np.array([[0.9975641, 0.0000000, 0.0697565],
        #                                 [0.0000000, 1.0000000, 0.0000000],
        #                                 [-0.0697565, 0.0000000, 0.9975641]])
        # static_trans[0, 3] = 0.15
        # static_trans[2, 3] = -0.05
        static_trans[:3,:3] = np.array([[1.0000000, 0.0000000, 0.0000000],
                                        [0.0000000, 0.9975641, -0.0697565],
                                        [0.0000000, 0.0697565, 0.9975641]])
        static_trans[1, 3] = -0.05
        static_trans[2, 3] = 0.15

        return np.matmul(pose, static_trans)
    
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
    
    def _draw_progress_bar(self, percent, bar_len=20):
        # percent float from 0 to 1.
        sys.stdout.write("\r")
        sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(bar_len * percent), bar_len, percent * 100))
        sys.stdout.flush()
    
    def plot_positions(self):
        assert len(self._frames) != 0

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('equal')

        # Plotting associated poses to each frame
        x_coords = [frame.pose[0, 3] for frame in self._frames]
        y_coords = [frame.pose[1, 3] for frame in self._frames]
        z_coords = [frame.pose[2, 3] for frame in self._frames]
        ax.plot3D(x_coords, y_coords, z_coords, 'r')    

        # Plotting original poses
        # Get pose info
        x_orig = [float(self._poses[tstamp][6]) for tstamp in self._pose_timestamps]
        y_orig = [float(self._poses[tstamp][7]) for tstamp in self._pose_timestamps]
        z_orig = [float(self._poses[tstamp][8]) for tstamp in self._pose_timestamps]
        ax.plot3D(x_orig, y_orig, z_orig, 'b')

        ax.set_xlabel('X(m)')
        ax.set_ylabel('Y(m)')
        ax.set_zlabel('Z(m)')

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        ax.view_init(21,-165)
        plt.legend(loc=1, fontsize=12)

        plt.show()
    
    def fuse(self):
        # Estimate voxel volume bounds
        print("Estimating voxel volume bounds...")        
        
        vol_bnds = np.zeros((3,2))
        for frame in self._frames:
            # Read depth image and camera pose
            depth_im = frame.load_D()
            cam_pose = frame.pose

            # Compute camera view frustum and extend convex hull
            view_frust_pts = fusion.get_view_frustum(depth_im, self._cam_intr, cam_pose)
            vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
        
        # Initialize voxel volume
        print("Initializing voxel volume...")
        self._tsdf_vol = fusion.TSDFVolume(vol_bnds, self._voxel_size)

        # Loop through RGB-D images and fuse them together
        t0_elapse = time.time()
        for i,frame in enumerate(self._frames):
            #print("Fusing frame %d/%d" % (i+1, len(self._frames)))
            self._draw_progress_bar(float(i) / len(self._frames))

            # Read RGB-D image and camera pose
            color_image = frame.load_RGB()
            depth_im = frame.load_D()
            cam_pose = frame.pose

            # Integrate observation into voxel volume (assume color aligned with depth)
            self._tsdf_vol.integrate(color_image, depth_im, self._cam_intr
            , cam_pose, obs_weight=1.)
        
        self._draw_progress_bar(1.0)
        
        fps = len(self._frames) / (time.time() - t0_elapse)
        print("Average FPS: {:.2f}".format(fps))
    
    def save_mesh(self):
        # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
        now = datetime.datetime.now()
        fullday = now.strftime("%Y%m%d")
        fullhour = now.strftime("%H_%M_%S")
        
        mesh_filename = os.path.join(self._data_dir, 'mesh_' + fullday + '_' + fullhour + '.ply')
        print("Saving mesh to %s ..." % mesh_filename)
        verts, faces, norms, colors = self._tsdf_vol.get_mesh()
        fusion.meshwrite(mesh_filename, verts, faces, norms, colors)

if __name__ == "__main__":
    # Reading parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="directory where the files are stored")
    parser.add_argument("-vs", "--voxel_size", default=0.02,
                        help="voxel size in meters")
    parser.add_argument("-pose", "--pose_filename", default='_worldpose_global_pose.csv',
                        help="pose filename where MAV absolute positions are saved")
    parser.add_argument("-cam", "--caminfo_filename",default='_camera_color_camera_info.csv', help="camera info filename where calibration info is saved")
    args = parser.parse_args()

    # Assessing if the data directory exists
    if (not os.path.exists(args.data_dir)):
        print("Directory does not exist!")
        sys.exit(0)
    
    builder = TSDFBuilder(args)
    builder.fuse()
    builder.save_mesh()
    
    #builder.plot_positions()