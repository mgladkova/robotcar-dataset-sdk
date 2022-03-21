import open3d
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize LMS scan data')
    parser.add_argument('--scan_path', type=str, help='Location of a .bin file')

    args = parser.parse_args()

    scan_file = open(args.scan_path, 'rb')
    scan = np.fromfile(scan_file, np.float64)
    scan_file.close()

    scan = scan.reshape((3, len(scan) // 3)).transpose()

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.ascontiguousarray(np.asarray(scan).astype(np.float64)))
    open3d.visualization.draw_geometries([pcd])
