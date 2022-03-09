################################################################################
# Adapted from Oxford robotcar-dataset sdk project_lasert_into_camera.py
################################################################################

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import open3d
import copy

from build_pointcloud import build_pointcloud_distance_range_masked
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

def load_transforms(model, extrinsics_dir, poses_file):
    extrinsics_path = os.path.join(extrinsics_dir, model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    G_camera_vehicle = build_se3_transform(extrinsics)

    G_camera_posesource = None
    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)
    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
    else:
        # VO frame and vehicle frame are the same
        G_camera_posesource = G_camera_vehicle

    return G_camera_vehicle, G_camera_posesource

def load_timestamps(timestamps_path):
    timestamps = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            timestamps.append(int(line.split(' ')[0]))
    return timestamps

def filter_pointcloud(xyzw, rflct, model, image_shape):
    # find which points lie in front of the camera
    in_front = [i for i in range(0, xyzw.shape[1]) if xyzw[2, i] > 0.1]
    xyzw = xyzw[:, in_front]
    rflct = rflct[in_front]

    # find which points lie in the image range
    uv = np.vstack((model.focal_length[0] * xyzw[0, :] / xyzw[2, :] + model.principal_point[0],
                    model.focal_length[1] * xyzw[1, :] / xyzw[2, :] + model.principal_point[1]))

    # mask points in the frustum (front & image range) of the camera
    u, v, depth = np.ravel(uv[0, :]), np.ravel(uv[1, :]), np.ravel(xyzw[2, :])
    mask_depth = ((depth > 0) & (depth < 70))
    mask_image = ((u >= 0.5) & (u < image_shape[1]) & (v >= 0.5) & \
                (v < image_shape[0]))
    mask = (mask_depth & mask_image)

    return xyzw[:3, mask]

def remove_ground_plane(pointcloud):
    # create Pointcloud for processing
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(
        -np.ascontiguousarray(pointcloud.transpose().astype(np.float64)))
    # fit a ground plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3,
                                            ransac_n=3,
                                            num_iterations=3000)
    plane_normal = np.array(plane_model[:3])
    plane_normal /= np.linalg.norm(plane_normal)
    print("Plane normal ", plane_normal)

    # remove points on the plane
    downpointcloud = pcd.select_by_index(inliers, invert=True)

    # remove points below the plane
    downpoints = np.asarray(downpointcloud.points)
    distance_ground = np.dot(downpoints, plane_normal)
    # camera is located ~1.36 meters above the ground
    # so if plane passes through the camera origin then points below -1.36 should be below the ground
    above_ground_idx = [i for i in range(len(distance_ground)) if distance_ground[i] > -1.3]
    return downpointcloud.select_by_index(above_ground_idx)

def downsample_pointcloud(pointcloud, num_points):
    vsize = 0.5
    downpointcloud = pointcloud.voxel_down_sample(voxel_size=vsize)

    # if the downsampled pointcloud is too small
    count = 1
    vsize_new = vsize
    while len(downpointcloud.points) < num_points*0.9 and count < 10:
        vsize_new = vsize_new - 0.05
        downpointcloud = pointcloud.voxel_down_sample(voxel_size=vsize_new)
        count += 1

    if vsize != vsize_new:
        vsize = vsize_new

    # if the downsampled pointcloud is too large
    count = 1
    vsize_new = vsize
    while len(downpointcloud.points) > num_points*1.1 and count < 10:
        vsize_new = vsize_new + 0.05
        downpointcloud = pointcloud.voxel_down_sample(voxel_size=vsize_new)
        count += 1

    if vsize != vsize_new:
        vsize = vsize_new

    print("Final grid size = ", vsize)

    return downpointcloud

def transform_image_laser(downpoints, model, extrinsics_dir, poses_file):
    _, G_camera_posesource = load_transforms(model, extrinsics_dir, poses_file)
    downpoints_homog = np.vstack((downpoints.T, np.ones((1, downpoints.shape[0]))))
    downpoints_camera = np.linalg.solve(np.linalg.inv(model.G_camera_image), downpoints_homog)
    downpoints_posesource = np.dot(np.linalg.inv(G_camera_posesource), downpoints_camera)

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', args.laser_dir).group(0)
    with open(os.path.join(args.extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses_type = re.search('(vo|ins|rtk)\.csv', args.poses_file).group(1)
    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(args.extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser)

    downpoints_laser = np.dot(np.linalg.inv(G_posesource_laser), downpoints_posesource)
    downpoints_laser = downpoints_laser[:3, :]
    return downpoints_laser

def normalize_data(pointcloud):
    downpoints_laser = copy.copy(pointcloud)
    # centralize and normalize data to range [-1, 1]
    centroid = np.mean(downpoints_laser, axis=1)
    downpoints_laser -= centroid

    downpoints_laser_min = np.min(downpoints_laser, axis=1)
    downpoints_laser_ptp = np.ptp(downpoints_laser, axis=1)
    downpoints_laser = 2.*(downpoints_laser - downpoints_laser_min) / downpoints_laser_ptp - 1
    return downpoints_laser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--laser_dir', type=str, help='Directory containing LIDAR scans')
    parser.add_argument('--mask_dir', type=str, help='Directory containing masks to remove movable objects from the scans')
    parser.add_argument('--poses_file', type=str, help='File containing either INS or VO poses')
    parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
    parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
    parser.add_argument('--image_idx', type=int, default=0, help='Index of image to associate velodyne pointcloud to')
    parser.add_argument('--num_points', type=int, default=4500, help='Number of points in each pointcloud')
    parser.add_argument('--visualize_ptcld', action='store_true')
    parser.add_argument('--visualize_image', action='store_true')

    args = parser.parse_args()

    model = CameraModel(args.models_dir, args.image_dir)
    _, G_camera_posesource = load_transforms(model, args.extrinsics_dir, args.poses_file)

    timestamps_path = os.path.join(args.image_dir, os.pardir, model.camera + '.timestamps')
    if not os.path.isfile(timestamps_path):
        timestamps_path = os.path.join(args.image_dir, os.pardir, os.pardir, model.camera + '.timestamps')

    timestamps = load_timestamps(timestamps_path)
    image_idx = args.image_idx
    timestamp = timestamps[image_idx] if ((image_idx >= 0) and (image_idx < len(timestamps))) else timestamps[0]

    print("Image timestamp ", timestamp)
    distance = 20
    ptcld, rflct = build_pointcloud_distance_range_masked(args.laser_dir, args.poses_file, args.extrinsics_dir, timestamp, G_camera_posesource, model, \
                                                          distance, timestamps, args.mask_dir)

    # transform velodyne pointcloud to camera coordinate system
    xyz = np.dot(G_camera_posesource, ptcld)
    if xyz.shape[0] == 3:
        xyz = np.stack((xyz, np.ones((1, xyz.shape[1]))))
    xyzw = np.linalg.solve(model.G_camera_image, xyz)

    image_path = os.path.join(args.image_dir, str(timestamp) + '.png')
    image = load_image(image_path, model)

    pointcloud = filter_pointcloud(xyzw, rflct, model, image.shape)

    noplane_cloud = remove_ground_plane(pointcloud)

    downpointcloud = downsample_pointcloud(noplane_cloud, args.num_points)
    downpoints = np.asarray(downpointcloud.points)

    print("Number of downsampled points = ", downpoints.shape)

    if args.visualize_image:
        depth_downsmpl = downpoints[:, 2]
        uv_downsmpl = np.vstack((model.focal_length[0] * downpoints[:, 0] / depth_downsmpl + model.principal_point[0],
                        model.focal_length[1] * downpoints[:, 1] / depth_downsmpl + model.principal_point[1]))

        plt.imshow(image)
        plt.scatter(np.ravel(uv_downsmpl[0, :]), np.ravel(uv_downsmpl[1, :]), s=2, c=depth_downsmpl, edgecolors='none', cmap='jet')
        plt.xlim(0, image.shape[1])
        plt.ylim(image.shape[0], 0)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    # transform to velodyne coordinate frame for the output
    downpoints_laser = transform_image_laser(downpoints, model, args.extrinsics_dir, args.poses_file)
    downpoints_laser = normalize_data(downpoints_laser)

    if args.visualize_ptcld:
        downpcd = open3d.geometry.PointCloud()
        downpcd.points = open3d.utility.Vector3dVector(-np.ascontiguousarray(np.asarray(downpoints_laser.T).astype(np.float64)))
        open3d.visualization.draw_geometries([downpcd])