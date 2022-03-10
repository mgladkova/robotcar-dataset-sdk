################################################################################
# Adapted from Oxford robotcar-dataset sdk project_lasert_into_camera.py
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import open3d

from build_pointcloud import build_pointcloud_distance_range_masked
from image import load_image
from camera_model import CameraModel

from create_image_pointcloud_pairs import load_transforms, load_timestamps, filter_pointcloud, \
                                            remove_ground_plane, downsample_pointcloud, normalize_data, transform_image_laser

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
    downpoints_laser = transform_image_laser(downpoints, model, args.extrinsics_dir, args.laser_dir, args.poses_file)
    downpoints_laser = normalize_data(downpoints_laser)

    if args.visualize_ptcld:
        downpcd = open3d.geometry.PointCloud()
        downpcd.points = open3d.utility.Vector3dVector(np.ascontiguousarray(np.asarray(downpoints_laser.T).astype(np.float64)))
        open3d.visualization.draw_geometries([downpcd])