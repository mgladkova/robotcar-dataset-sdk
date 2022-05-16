################################################################################
# Adapted from Oxford robotcar-dataset sdk create_image_pointcloud_pairs.py
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import copy

from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

from create_image_pointcloud_pairs import load_image_rtk

parser = argparse.ArgumentParser(
    description='Filter positive query samples by projection')
parser.add_argument('--pickle_path', type=str, help='Pickle file path')
parser.add_argument('--data_dir', type=str, help='Directory containing data')
parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--save', action='store_true')

args = parser.parse_args()
queries = None
with open(args.pickle_path, 'rb') as handle:
    queries = pickle.load(handle)

image_rtk_poses_all = {}
image_rtk_dir = os.path.join(args.data_dir, "image_rtk_new")
for subdir in os.listdir(image_rtk_dir):
    image_rtk_pos_path = os.path.join(image_rtk_dir, subdir, "image_rtk_full.csv")
    img_rtk_poses_pos = load_image_rtk(image_rtk_pos_path, full=True)
    image_rtk_poses_all[subdir] = img_rtk_poses_pos

extrinsics_path = os.path.join(args.extrinsics_dir, 'stereo.txt')
with open(extrinsics_path) as extrinsics_file:
    extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
G_camera_vehicle = build_se3_transform(extrinsics)

with open(os.path.join(args.extrinsics_dir, 'lms_front.txt')) as extrinsics_file:
    extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

with open(os.path.join(args.extrinsics_dir, 'ins.txt')) as extrinsics_file:
    extrinsics = next(extrinsics_file)
    extr = build_se3_transform([float(x) for x in extrinsics.split(' ')])
    G_camera_posesource = G_camera_vehicle * extr
    G_posesource_laser = np.linalg.solve(extr, G_posesource_laser)

outfile = None
if args.save:
    outfile = args.pickle_path[:-7] + "_filtered.pickle"

filtered_queries = {}
for i in range(len(queries)):
    dict_0 = queries[i]
    qname = os.path.join(args.data_dir, dict_0['query'])
    positive_names = []
    seq_dirs = []
    orig_positives = []
    for pos in dict_0['positives']:
        sub = queries[pos]['query'][dict_0['query'].rfind('/201') + 1:]
        seq_dirs.append(sub[:sub.find('/')])
        positive_names.append(os.path.join(args.data_dir, queries[pos]['query']))
        orig_positives.append(pos)

    seq_query = dict_0['query'][dict_0['query'].rfind('/201') + 1:]
    seq_query = seq_query[:seq_query.find('/')]
    timestamp_query = dict_0['query'][dict_0['query'].rfind('/') + 1 : dict_0['query'].rfind('.')]
    print("Index ", i, " query seq ", seq_query, " timestamp ", timestamp_query)

    image_dir = os.path.join(args.data_dir, seq_query, "stereo", "centre")

    image_path =  os.path.join(image_dir, timestamp_query + ".png")
    model = CameraModel(args.models_dir, image_dir)

    orig_points = np.fromfile(qname)
    orig_points = np.reshape(orig_points, (3, -1))
    pointcloud_posesource = np.dot(G_posesource_laser, np.vstack([orig_points, np.ones((1, orig_points.shape[1]))]))
    pointcloud_camera = np.dot(G_camera_posesource, pointcloud_posesource)

    image= load_image(image_path, model)
    if args.visualize:
        uv, depth = model.project(pointcloud_camera, image.shape)

        plt.imshow(image)
        plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]),
                    s=2, c=depth, edgecolors='none', cmap='jet')
        plt.xlim(0, image.shape[1])
        plt.ylim(image.shape[0], 0)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    filtered_positives = []
    for j in range(len(seq_dirs)):
        seq_dir = seq_dirs[j]
        posname = positive_names[j]
        timestamp = posname[posname.rfind('/') + 1:-4]

        pointcloud = np.linalg.solve(image_rtk_poses_all[seq_dir][int(timestamp)], np.dot(image_rtk_poses_all[seq_query][int(timestamp_query)], pointcloud_posesource))
        pointcloud = np.dot(G_camera_posesource, pointcloud)

        if args.visualize:
            image_path = os.path.join(args.data_dir, seq_dir, "stereo", "centre", timestamp + ".png")
            image= load_image(image_path, model)
        uv, depth = model.project(pointcloud, image.shape)

        if uv.shape[1] > 0.3 * pointcloud_posesource.shape[1]:
            filtered_positives.append(orig_positives[j])

        if args.visualize:
            plt.imshow(image)
            plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]),
                        s=2, c=depth, edgecolors='none', cmap='jet')
            plt.xlim(0, image.shape[1])
            plt.ylim(image.shape[0], 0)
            plt.xticks([])
            plt.yticks([])
            plt.show()
    print("\tFiltered size: {} / {}".format(len(filtered_positives), len(dict_0['positives'])))
    filtered_queries[i] = {"query": dict_0['query'],
                           "positives": filtered_positives,
                           "negatives": dict_0['negatives']}

if args.save and not outfile is None:
    with open(outfile, 'wb') as handle:
        pickle.dump(filtered_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
