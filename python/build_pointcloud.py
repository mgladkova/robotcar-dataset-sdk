################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import os
import re
from time import time
import numpy as np
import cv2
import pickle

from transform import build_se3_transform
from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from velodyne import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud

def build_pointcloud(lidar_dir, poses_file, extrinsics_dir, start_time, end_time, origin_time=-1):
    """Builds a pointcloud by combining multiple LIDAR scans with odometry information.

    Args:
        lidar_dir (str): Directory containing LIDAR scans.
        poses_file (str): Path to a file containing pose information. Can be VO or INS data.
        extrinsics_dir (str): Directory containing extrinsic calibrations.
        start_time (int): UNIX timestamp of the start of the window over which to build the pointcloud.
        end_time (int): UNIX timestamp of the end of the window over which to build the pointcloud.
        origin_time (int): UNIX timestamp of origin frame. Pointcloud coordinates are relative to this frame.

    Returns:
        numpy.ndarray: 3xn array of (x, y, z) coordinates of pointcloud
        numpy.array: array of n reflectance values or None if no reflectance values are recorded (LDMRS)

    Raises:
        ValueError: if specified window doesn't contain any laser scans.
        IOError: if scan files are not found.

    """
    if origin_time < 0:
        origin_time = start_time

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', lidar_dir).group(0)
    timestamps_path = os.path.join(lidar_dir, os.pardir, lidar + '.timestamps')

    timestamps = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            if start_time <= timestamp <= end_time:
                timestamps.append(timestamp)

    if len(timestamps) == 0:
        raise ValueError("No LIDAR data in the given time bracket.")

    print(lidar)

    with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)

    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser)

        poses = interpolate_ins_poses(poses_file, timestamps, origin_time, use_rtk=(poses_type == 'rtk'))
    else:
        # sensor is VO, which is located at the main vehicle frame
        poses = interpolate_vo_poses(poses_file, timestamps, origin_time)

    pointcloud = np.array([[0], [0], [0], [0]])
    if lidar == 'ldmrs':
        reflectance = None
    else:
        reflectance = np.empty((0))

    #print("Number of poses = ", len(poses))
    for i in range(0, len(poses)):
        scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.bin')
        if "velodyne" not in lidar:
            if not os.path.isfile(scan_path):
                continue

            with open(scan_path, "rb") as f:
                scan = pickle.load(f)

            # scan_file = open(scan_path)
            # scan = np.fromfile(scan_file, np.float64)
            # scan_file.close()
            scan = scan.reshape((int(len(scan) / 3), 3)).transpose()

            if lidar != 'ldmrs':
                # LMS scans are tuples of (x, y, reflectance)
                reflectance = np.concatenate((reflectance, np.ravel(scan[2, :])))
                scan[2, :] = np.zeros((1, scan.shape[1]))
        else:
            if os.path.isfile(scan_path):
                ptcld = load_velodyne_binary(scan_path)
            else:
                scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.png')
                if not os.path.isfile(scan_path):
                    continue
                ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(scan_path)
                ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)

            reflectance = np.concatenate((reflectance, ptcld[3]))
            scan = ptcld[:3]

        scan = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))])) # dot(4x4, 4xN)
        pointcloud = np.hstack([pointcloud, scan])


    pointcloud = pointcloud[:, 1:]
    if pointcloud.shape[1] == 0:
        raise IOError("Could not find scan files for given time range in directory " + lidar_dir)

    return pointcloud, reflectance


def build_pointcloud_distance_range_masked(lidar_dir, poses_file, extrinsics_dir, origin_time, G_camera_posesource, model, \
                                            distance=20, min_point_number=4500, img_timestamps=None, mask_dir=None):
    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', lidar_dir).group(0)
    timestamps_path = os.path.join(lidar_dir, os.pardir, lidar + '.timestamps')

    #hopefully enough to cover 50 m range (might not work in case of long stays)
    start_time = origin_time
    end_time = origin_time + 5e8
    timestamps = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            if start_time <= timestamp <= end_time:
                timestamps.append(timestamp)

    if len(timestamps) == 0:
        raise ValueError("No LIDAR data in the given time bracket.")

    with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)

    img_poses = None
    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser)

        poses = interpolate_ins_poses(poses_file, timestamps, origin_time, use_rtk=(poses_type == 'rtk'))
        if not img_timestamps is None and not mask_dir is None:
            img_poses = interpolate_ins_poses(poses_file, img_timestamps, origin_time, use_rtk=(poses_type == 'rtk'))
    else:
        # sensor is VO, which is located at the main vehicle frame
        poses = interpolate_vo_poses(poses_file, timestamps, origin_time)
        if not img_timestamps is None and not mask_dir is None:
            img_poses = interpolate_vo_poses(poses_file, img_timestamps, origin_time)

    pointcloud = np.array([[0], [0], [0], [0]])
    if lidar == 'ldmrs':
        reflectance = None
    else:
        reflectance = np.empty((0))

    orig_pose = poses[0]
    count = 0
    for i in range(0, len(poses)):
        diff_pose = np.linalg.inv(orig_pose) @ poses[i] # convention: camera-world
        diff_pose_norm = np.linalg.norm(diff_pose[:3, 3])
        if diff_pose_norm > distance:
            break

        scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.bin')
        #scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.npy')
        reflectance_current = None
        if "velodyne" not in lidar:
            if not os.path.isfile(scan_path):
                continue

            scan_file = open(scan_path, 'rb')
            #start = time()
            #scan = np.load(scan_file)
            scan = np.fromfile(scan_file, np.double)
            # np.save(scan_file_new, scan)
            scan_file.close()
            #end = time()
            # print("Loading scan {} seconds".format(end - start))

            scan = scan.reshape((len(scan) // 3, 3)).transpose()

            if lidar != 'ldmrs':
                # LMS scans are tuples of (x, y, reflectance)
                reflectance_current = np.ravel(scan[2, :])
                # reflectance = np.concatenate((reflectance, np.ravel(scan[2, :])))
                scan[2, :] = np.zeros((1, scan.shape[1]))
        else:
            if os.path.isfile(scan_path):
                ptcld = load_velodyne_binary(scan_path)
            else:
                scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.png')
                if not os.path.isfile(scan_path):
                    continue
                ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(scan_path)
                ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)

            reflectance_current = ptcld[3]
            scan = ptcld[:3]

            #removes points on the ego-car (x-axis points backward), visible car part ~ 3.5 m
            mask = (scan[0] < -4)
            reflectance_current = reflectance_current[mask]
            mask = np.broadcast_to(mask, scan.shape)
            scan = np.reshape(scan[mask], (3, -1))

        scan_laser = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))])) # dot(4x4, 4xN)
        remove_indices = np.array([], dtype=int)
        if not img_poses is None and not mask_dir is None:
            img_pose = None
            img_timestamp = None
            kernel = np.ones((15, 15), 'uint8')
            for j in range(1, len(img_timestamps)):
                if img_timestamps[j - 1] <= timestamps[i] <= img_timestamps[j]:
                    left_diff = timestamps[i] - img_timestamps[j - 1]
                    right_diff = img_timestamps[j] - timestamps[i]
                    if left_diff < right_diff and left_diff < 1e5:
                        img_pose = img_poses[j - 1]
                        img_timestamp = img_timestamps[j - 1]
                    elif right_diff < 1e5:
                        img_pose = img_poses[j]
                        img_timestamp = img_timestamps[j]
                    break

            if not img_pose is None:
                scan_camera = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))])) # dot(4x4, 4xN)
                scan_camera = np.dot(G_camera_posesource, scan_camera)
                scan_image = np.linalg.solve(model.G_camera_image, scan_camera)

                image_path = os.path.join(mask_dir, str(img_timestamp) + '.png')
                image = cv2.imread(image_path, 0)
                image = cv2.dilate(image, kernel)

                # cv2.imshow("Mask", image)
                # cv2.waitKey(5)

                points = np.vstack((model.focal_length[0] * scan_image[0, :] / scan_image[2, :] + model.principal_point[0],
                                model.focal_length[1] * scan_image[1, :] / scan_image[2, :] + model.principal_point[1]))
                for j in range(points.shape[1]):
                    if 0.5 <= points[0, j] <= image.shape[1] and \
                       0.5 <= points[1, j] <= image.shape[0] and \
                       image[int(points[1, j]), int(points[0, j])]:
                        remove_indices = np.append(remove_indices, j)
        if remove_indices.shape[0] > 0:
            scan_laser = np.delete(scan_laser, remove_indices, axis=1)
            reflectance_current = np.delete(reflectance_current, remove_indices)

        pointcloud = np.hstack([pointcloud, scan_laser])
        reflectance = np.concatenate((reflectance, reflectance_current))
        count += 1

    #print("Number of poses = ", count)

    pointcloud = pointcloud[:, 1:]
    if pointcloud.shape[1] == 0:
        raise IOError("Could not find scan files for given time range in directory " + lidar_dir)

    return pointcloud, reflectance


if __name__ == "__main__":
    import argparse
    import open3d

    parser = argparse.ArgumentParser(description='Build and display a pointcloud')
    parser.add_argument('--poses_file', type=str, default=None, help='File containing relative or absolute poses')
    parser.add_argument('--extrinsics_dir', type=str, default=None,
                        help='Directory containing extrinsic calibrations')
    parser.add_argument('--laser_dir', type=str, default=None, help='Directory containing LIDAR data')

    args = parser.parse_args()

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', args.laser_dir).group(0)
    timestamps_path = os.path.join(args.laser_dir, os.pardir, lidar + '.timestamps')
    print(timestamps_path)
    timestamps = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            entries = line.rstrip().split()
            timestamps.append(int(entries[0]))

    start_time = timestamps[5000]
    end_time = timestamps[5100] #start_time + 1e7

    pointcloud, reflectance = build_pointcloud(args.laser_dir, args.poses_file,
                                               args.extrinsics_dir, start_time, end_time)

    if reflectance is not None:
        colours = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min())
        colours = 1 / (1 + np.exp(-10 * (colours - colours.mean())))
    else:
        colours = 'gray'

    # Pointcloud Visualisation using Open3D
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name=os.path.basename(__file__))
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
    render_option.point_color_option = open3d.visualization.PointColorOption.ZCoordinate
    coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(coordinate_frame)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(
        -np.ascontiguousarray(pointcloud[[1, 0, 2]].transpose().astype(np.float64)))
    if reflectance is not None:
        pcd.colors = open3d.utility.Vector3dVector(np.tile(colours[:, np.newaxis], (1, 3)).astype(np.float64))
    # Rotate pointcloud to align displayed coordinate frame colouring
    #pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))
    vis.add_geometry(pcd)
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = build_se3_transform([0, 3, 10, 0, -np.pi * 0.42, -np.pi / 2])
    view_control.convert_from_pinhole_camera_parameters(params)
    vis.run()
