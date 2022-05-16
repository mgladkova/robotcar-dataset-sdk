################################################################################
# Adapted from Oxford robotcar-dataset sdk project_lasert_into_camera.py
################################################################################

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import open3d


from sklearn.neighbors import KDTree
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from build_pointcloud import build_pointcloud_distance_range_masked
from image import load_image
from camera_model import CameraModel

from create_image_pointcloud_pairs import load_transforms, load_timestamps, filter_pointcloud, \
                                            remove_ground_plane, downsample_pointcloud, normalize_data, \
                                            transform_image_laser, load_image_rtk, \
                                            filter_positive_pairs_by_projection

from time import time

def build_kd_tree(submaps_dir, current_seq):
    df_pc = pd.DataFrame(columns=['timestamp', 'northing', 'easting'])
    for subdir in os.listdir(submaps_dir):
        if subdir == current_seq:
            continue

        df_pc_locations = pd.read_csv(os.path.join(submaps_dir, subdir, "pointcloud_30m_15overlap_new.csv"), sep=',')
        df_pc_locations['timestamp'] = subdir + "/" + df_pc_locations['timestamp'].astype(str) + '.bin'

        for index, row in df_pc_locations.iterrows():
            df_pc = df_pc.append(row, ignore_index=True)

    return df_pc, KDTree(df_pc[['northing', 'easting']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--submaps_dir', type=str, help='Directory containing submaps (first script run pass)')
    parser.add_argument('--laser_dir', type=str, help='Directory containing LIDAR scans')
    parser.add_argument('--mask_dir', type=str, help='Directory containing masks to remove movable objects from the scans')
    parser.add_argument('--poses_file', type=str, help='File containing either INS or VO poses')
    parser.add_argument('--image_rtk', type=str, help='File containing either RTK poses for image timestamps')
    parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
    parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
    parser.add_argument('--image_idx', type=int, default=0, help='Index of image to associate velodyne pointcloud to')
    parser.add_argument('--num_points', type=int, default=4500, help='Number of points in each pointcloud')
    parser.add_argument('--visualize_ptcld', action='store_true')
    parser.add_argument('--visualize_image', action='store_true')

    args = parser.parse_args()

    image_dir = args.image_dir
    submaps_dir = args.submaps_dir
    poses_file = args.poses_file
    laser_dir = args.laser_dir

    dataset_dir = laser_dir[:laser_dir.find('/20')]
    current_seq = laser_dir[laser_dir.find('/20') + 1:]
    current_seq = current_seq[:current_seq.find('/')]
    print("Dataset directory ", dataset_dir)
    print("Current sequence ", current_seq)

    assert('full' in args.image_rtk)

    df_submaps, kdtree = build_kd_tree(args.submaps_dir, current_seq)
    print("Built a KD-Tree")

    img_rtk_poses = load_image_rtk(args.image_rtk, full=True)
    print("Loaded {} image poses".format(len(img_rtk_poses)))

    model = CameraModel(args.models_dir, image_dir)
    start = time()
    _, G_camera_posesource = load_transforms(model, args.extrinsics_dir, poses_file)
    end = time()
    print("Load transforms took {} seconds".format(end - start))

    timestamps_path = os.path.join(image_dir, os.pardir, model.camera + '.timestamps')
    if not os.path.isfile(timestamps_path):
        timestamps_path = os.path.join(image_dir, os.pardir, os.pardir, model.camera + '.timestamps')

    start = time()
    timestamps = load_timestamps(timestamps_path)
    end = time()
    print("Load timestamps took {} seconds".format(end - start))
    image_idx = args.image_idx
    timestamp = timestamps[image_idx] if ((image_idx >= 0) and (image_idx < len(timestamps))) else timestamps[0]

    print("Image timestamp ", timestamp)

    print(img_rtk_poses[timestamp][:2, 3].flatten())
    ind_nn = kdtree.query_radius(img_rtk_poses[timestamp][:2, 3].flatten(), r=10)
    print("Number of positives within radius of 10 m = ", len(ind_nn[0]))

    distance = 30
    start = time()
    ptcld, rflct = build_pointcloud_distance_range_masked(laser_dir, poses_file, args.extrinsics_dir, timestamp, G_camera_posesource, model, \
                                                          distance=distance, img_timestamps=timestamps, mask_dir=args.mask_dir, min_point_number=args.num_points)
    end = time()
    print("Build pointcloud took {} seconds".format(end - start))

    # ptcld, rflct = build_pointcloud(laser_dir, poses_file, args.extrinsics_dir, timestamp, timestamp + 1e7)

    # transform velodyne pointcloud to camera coordinate system
    xyz = np.dot(G_camera_posesource, ptcld)
    if xyz.shape[0] == 3:
        xyz = np.stack((xyz, np.ones((1, xyz.shape[1]))))
    xyzw = np.linalg.solve(model.G_camera_image, xyz)

    image_path = os.path.join(image_dir, str(timestamp) + '.png')
    image = load_image(image_path, model)

    start = time()
    pointcloud = filter_pointcloud(xyzw, rflct, model, image.shape)
    end = time()
    print("Filter pointcloud took {} seconds".format(end - start))

    start = time()
    noplane_cloud = remove_ground_plane(pointcloud)
    end = time()
    print("Remove ground plane took {} seconds".format(end - start))

    #downpointcloud = noplane_cloud
    start = time()
    downpointcloud = downsample_pointcloud(noplane_cloud, args.num_points)
    downpoints = np.asarray(downpointcloud.points)
    end = time()
    print("Downsampling took {} seconds".format(end - start))

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

    print(img_rtk_poses[timestamp])
    filtered_ind_nn = filter_positive_pairs_by_projection(downpoints, img_rtk_poses[timestamp], dataset_dir, df_submaps, \
                                                          ind_nn[0], model, G_camera_posesource, image.shape)
    print("Filtered number of positive pairs: ", len(filtered_ind_nn))


    # transform to velodyne coordinate frame for the output
    downpoints_laser = transform_image_laser(downpoints, model, args.extrinsics_dir, laser_dir, poses_file)
    downpoints_laser = normalize_data(downpoints_laser)

    if args.visualize_ptcld:
        downpcd = open3d.geometry.PointCloud()
        downpcd.points = open3d.utility.Vector3dVector(np.ascontiguousarray(np.asarray(downpoints_laser.T).astype(np.float64)))
        open3d.visualization.draw_geometries([downpcd])