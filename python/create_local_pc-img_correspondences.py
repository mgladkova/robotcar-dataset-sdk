import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from sklearn.neighbors import KDTree

from transform import build_se3_transform
from camera_model import CameraModel


def load_pc(pc_path, G_posesource_laser, G_camera_posesource):
    points_unnorm = np.reshape(np.fromfile(pc_path), (3, -1)).astype(np.float32)
    points_unnorm = np.concatenate((points_unnorm, np.ones((1, points_unnorm.shape[1]))), axis=0)

    points_cam = np.dot(np.dot(G_camera_posesource, G_posesource_laser), points_unnorm)
    return points_cam[:3, :]

def load_keypoint_detections(kpts_path):
    detections = None
    with open(kpts_path, 'r') as fin:
        next(fin, None) # skip first line with image size
        kpts_len = int(fin.readline().rstrip().split()[0])
        print("Number of keypoints ", kpts_len)
        detections = np.zeros((kpts_len, 2), dtype=int)
        for i, line in enumerate(fin):
            entries = line.rstrip().split()
            detections[i] = np.array([int(float(entries[0])), int(float(entries[1]))])
    return detections

def find_kpt_lidar_correspondences(pointcloud_uv, dets, img):
    print(pointcloud_uv.shape, dets.shape)
    kpts_tree = KDTree(dets)
    neigh, _ = kpts_tree.query_radius(pointcloud_uv, r=10, return_distance=True, sort_results=True)
    neighbors = np.zeros((pointcloud_uv.shape[0], 2), dtype=int)
    for i in range(len(neigh)):
        if len(neigh[i]) < 1:
            neighbors[i] = [-1, -1]
            continue
        neigh_idx = neigh[i][0]
        neighbors[i] = [dets[neigh_idx, 1], dets[neigh_idx, 0]]
    return neighbors.T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
    parser.add_argument('--image_dir', type=str, help='Undistorted image directory path')
    parser.add_argument('--kpts_dir', type=str, default=None, help='Keypoints directory path')
    parser.add_argument('--submap_path', type=str, help='Submap path')
    parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
    parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
    parser.add_argument('--outdir', type=str, help='Output directory')
    parser.add_argument('--color', action="store_true", help='Directory containing sensor extrinsics')

    args = parser.parse_args()

    # apart from camera intrinsics the rest is not relevant (e.g. distortion map, since input images should be already undistorted)
    model = CameraModel(args.models_dir, "stereo/centre")

    extrinsics_path = os.path.join(args.extrinsics_dir, model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
    G_camera_vehicle = build_se3_transform(extrinsics)

    with open(os.path.join(args.extrinsics_dir, 'lms_front.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    with open(os.path.join(args.extrinsics_dir, 'ins.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        T_vehicle_posesource = build_se3_transform([float(x) for x in extrinsics.split(' ')])
        G_posesource_laser = np.linalg.solve(T_vehicle_posesource, G_posesource_laser)
        G_camera_posesource = G_camera_vehicle * T_vehicle_posesource

    submap_path = args.submap_path
    pointcloud = load_pc(submap_path, G_posesource_laser, G_camera_posesource)

    seq_dir = submap_path[submap_path.rfind("/20") + 1:]
    seq_dir = seq_dir[:seq_dir.find("/")]
    timestamp = submap_path[submap_path.rfind("/") + 1:-3]
    image_path = os.path.join(args.image_dir, seq_dir, timestamp + "png")
    print("Loading corresponding image ", image_path)
    image = Image.open(image_path)
    image = np.array(image).astype(np.uint8)

    uv, depth = model.project(pointcloud, image.shape)
    u_coord = np.ravel(uv[0, :])
    v_coord = np.ravel(uv[1, :])
    rgb = np.reshape(image[v_coord.astype(int), u_coord.astype(int), :], (-1, 3)).T

    assert pointcloud.shape[1] == uv.shape[1]

    plt.imshow(image)

    if not args.kpts_dir is None:
        kpts_path = os.path.join(args.kpts_dir, seq_dir, timestamp + "txt")
        print("Loading keypoints from ", kpts_path)
        dets = load_keypoint_detections(kpts_path)
        print("Loaded ", dets.shape[0], " detections")
        vu = find_kpt_lidar_correspondences(uv.T, dets, image)

        plt.scatter(np.ravel(dets[:, 0]), np.ravel(dets[:, 1]), s=2)
    else:
        vu = uv[[1, 0], :]
        plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')

    ptcld_final = None
    if args.color:
        print("Saving pointcloud with color")
        ptcld_final = np.concatenate((pointcloud, rgb, vu), axis=0)
    else:
        ptcld_final = np.concatenate((pointcloud, vu), axis=0)
    print("Final pointcloud size = ", ptcld_final.shape)

    outsubdir = os.path.join(args.outdir, seq_dir, "pointcloud_50m_25overlap")
    if not os.path.exists(outsubdir):
        os.makedirs(outsubdir)

    ptcld_final.tofile(os.path.join(outsubdir, timestamp + "bin"))


    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.xticks([])
    plt.yticks([])
    plt.show()
