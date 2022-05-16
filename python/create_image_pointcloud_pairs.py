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
import csv
import sys

from build_pointcloud import build_pointcloud_distance_range_masked
from interpolate_poses import interpolate_ins_poses
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
            line = line.rstrip()
            entries = line.split()
            timestamp = int(entries[0])
            timestamps.append(timestamp)
    return timestamps

def load_timestamps_submaps_csv(submaps_csv_path):
    timestamps = []
    with open(submaps_csv_path, newline='') as csvfile:
        next(csvfile)
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            timestamp = int(row[0])
            timestamps.append(timestamp)
    return np.array(timestamps)

def filter_pointcloud(ptcld, rflct, model, image_shape):
    # find which points lie in front of the camera
    xyzw = copy.copy(ptcld)
    in_front = [i for i in range(0, xyzw.shape[1]) if xyzw[2, i] > 0]
    xyzw = xyzw[:, in_front]
    rflct = rflct[in_front]

    #print("In front of the camera ", np.count_nonzero(in_front))

    # find which points lie in the image range
    uv = np.vstack((model.focal_length[0] * xyzw[0, :] / xyzw[2, :] + model.principal_point[0],
                    model.focal_length[1] * xyzw[1, :] / xyzw[2, :] + model.principal_point[1]))

    # mask points in the frustum (front & image range) of the camera
    u, v = np.ravel(uv[0, :]), np.ravel(uv[1, :])
    mask_image = ((u >= 0.5) & (u < image_shape[1]) & (v >= 0.5) & \
                (v < image_shape[0]))

    #print("Within the image ", np.count_nonzero(mask_image))
    return xyzw[:3, mask_image]

def remove_ground_plane(pointcloud):
    # create Pointcloud for processing
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(
        np.ascontiguousarray(pointcloud.transpose().astype(np.float64)))

    if pointcloud.shape[1] < 100:
        return pcd

    #fit a ground plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1,
                                            ransac_n=3,
                                            num_iterations=1000)
    plane_normal = np.array(plane_model[:3])
    plane_normal /= np.linalg.norm(plane_normal)

    if plane_normal[1] > 0:
        plane_normal *= -1.

    threshold = -1.0
    yaxis = np.array([[0.], [-1.], [0.]], dtype=float)
    if np.arccos(np.dot(plane_normal, yaxis)) < 0.3:
        # remove points on the plane
        downpointcloud = pcd.select_by_index(inliers, invert=True)
        # remove points below the plane
        downpoints = np.asarray(downpointcloud.points)
    else:
        plane_normal = yaxis
        downpoints = copy.copy(pointcloud.transpose())
        downpointcloud = pcd
        #threshold = 0.5 # stricter threshold since we don't have exact ground plane normal

    #print(plane_normal)

    distance_ground = np.dot(downpoints, plane_normal)
    above_ground_idx = [i for i in range(len(distance_ground)) if distance_ground[i] > threshold]
    return downpointcloud.select_by_index(above_ground_idx)

def downsample_pointcloud(pointcloud, num_points):
    if np.asarray(pointcloud.points).shape[0] < num_points:
        return pointcloud

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

    return downpointcloud

def transform_image_laser(downpoints, model, extrinsics_dir, laser_dir, poses_file):
    _, G_camera_posesource = load_transforms(model, extrinsics_dir, poses_file)
    downpoints_homog = np.vstack((downpoints.T, np.ones((1, downpoints.shape[0]))))
    downpoints_camera = np.linalg.solve(np.linalg.inv(model.G_camera_image), downpoints_homog)
    downpoints_posesource = np.dot(np.linalg.inv(G_camera_posesource), downpoints_camera)

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', laser_dir).group(0)
    with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)
    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
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

def load_image_rtk(filepath, full=False):
    img_rtk_transl = {}
    with open(filepath, newline='') as csvfile:
        next(csvfile)
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            timestamp = int(row[0])
            if full:
                assert(len(row) == 7)
                xyzrpy = [float(x) for x in row[1:]]
                xyzrpy[-1] += np.pi * 1.5 # fix bug for yaw angle in RTK poses
                transl = build_se3_transform(xyzrpy)
            else:
                transl = np.array([float(row[1]), float(row[2])])
            img_rtk_transl[timestamp] = transl

    return img_rtk_transl

def filter_positive_pairs_by_projection(ptcld, rtk_pose, dataset_dir, submaps_all, orig_pos_indices, model, G_camera_posesource, image_shape):
    ptcld_hom = np.vstack((ptcld.T, np.ones((1, ptcld.shape[0]))))
    ptcld_posesource = np.linalg.solve(G_camera_posesource, np.dot(model.G_camera_image, ptcld_hom))
    ptcld_world = np.dot(rtk_pose, ptcld_posesource)
    reflect = np.ones((ptcld_world.shape[1], 1))
    filtered_indices = []
    for idx in orig_pos_indices:
        seq_dir, img_timestamp = submaps_all.iloc[idx]['timestamp'].split('/')
        img_timestamp = int(img_timestamp[:img_timestamp.rfind('.')])
        img_rtk_poses_file = os.path.join(dataset_dir, "image_rtk_new", seq_dir, "image_rtk_full.csv")

        img_rtk_poses = load_image_rtk(img_rtk_poses_file, full=True)
        ptcld_pos = np.linalg.solve(model.G_camera_image, np.dot(G_camera_posesource, np.linalg.solve(img_rtk_poses[img_timestamp], ptcld_world)))

        dummy = copy.copy(reflect)
        filtered_ptcld = filter_pointcloud(ptcld_pos, dummy, model, image_shape)
        filtered_ptcld = filtered_ptcld.T

        if filtered_ptcld.shape[0] > 0.3*ptcld_world.shape[1]:
            filtered_indices.append(idx)
    return filtered_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--laser_dir', type=str, help='Directory containing LIDAR scans')
    parser.add_argument('--mask_dir', type=str, help='Directory containing masks to remove movable objects from the scans')
    parser.add_argument('--poses_file', type=str, help='File containing either INS or VO poses')
    parser.add_argument('--image_rtk', type=str, help='File containing RTK absolute poses for image timestamps')
    parser.add_argument('--models_dir', type=str, default=None, help='Directory containing camera models')
    parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
    parser.add_argument('--output_dir', type=str, help='Output directory for submaps per image')
    parser.add_argument('--num_points', type=int, default=4500, help='Number of points in each pointcloud')
    parser.add_argument('--start_frame', type=int, default=0, help='Image index to start the collection from')
    parser.add_argument('--log', type=str, default='log.txt', help='Log file to write stdout to')

    args = parser.parse_args()

    sys.stdout = open(args.log, 'w')

    model = CameraModel(args.models_dir, args.image_dir)
    _, G_camera_posesource = load_transforms(model, args.extrinsics_dir, args.poses_file)

    timestamps_path = os.path.join(args.image_dir, os.pardir, model.camera + '.timestamps')
    if not os.path.isfile(timestamps_path):
        timestamps_path = os.path.join(args.image_dir, os.pardir, os.pardir, model.camera + '.timestamps')

    outdir = args.output_dir
    outsubdir = os.path.join(outdir, "pointcloud_50m_25overlap")
    if not os.path.exists(outsubdir):
        os.makedirs(outsubdir)
    csvfilename = os.path.join(outdir, "pointcloud_50m_25overlap.csv")

    submap_dist_thresh = 25
    submap_range = 50
    timestamps = load_timestamps(timestamps_path)
    start_idx = args.start_frame
    img_rtk_poses = load_image_rtk(args.image_rtk)
    prev_image_idx = -1
    print("Loaded {} image poses".format(len(img_rtk_poses)))
    with open(csvfilename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['timestamp', 'northing', 'easting'])

        for i in range(start_idx, len(timestamps)):
            timestamp = timestamps[i]
            if prev_image_idx >= 0:
                rel_transform = img_rtk_poses[timestamps[prev_image_idx]]- img_rtk_poses[timestamp]
                if np.linalg.norm(rel_transform) < submap_dist_thresh:
                    continue

            print("Image timestamp ", timestamp)
            ptcld, rflct = build_pointcloud_distance_range_masked(args.laser_dir, args.poses_file, args.extrinsics_dir, timestamp, G_camera_posesource, model, \
                                                                distance=submap_range, min_point_number=args.num_points)

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

            if downpoints.shape[0] < 1000:
                prev_image_idx = i
                continue

            # transform to velodyne coordinate frame for the output
            downpoints_laser = transform_image_laser(downpoints, model, args.extrinsics_dir, args.laser_dir, args.poses_file)
            #downpoints_laser = normalize_data(downpoints_laser)

            outfile = os.path.join(outsubdir, str(timestamp) + ".bin")
            downpoints_laser.tofile(outfile)
            prev_image_idx = i

            csvwriter.writerow([timestamp, *img_rtk_poses[timestamp]])
            sys.stdout.flush()

    sys.stdout.close()