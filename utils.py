import os
import numpy as np
import h5py
from pprint import pprint

def is_in_view(u, v, z_c, height, width):
    return (z_c>0 and u>=0 and u<width and v>=0 and v<height)

def create_depth_image(intrinsics, T_velo_to_cam, velo, height, width):
    reflectance = velo[:,3]
    points = velo[reflectance>0,:] # keep only lidar measurements with positive reflectance
    points[:,3] = 1 # convert the coordinates to homogeneous representation by setting 4th column to be all 1's
    coordinates_cam = np.dot(T_velo_to_cam, points.transpose()) # transform to camera frame
    N = coordinates_cam.shape[1]

    pixels_in = np.dot(intrinsics, coordinates_cam[:3,:])
    depth = np.zeros((height, width))
    for i in range(0, N):
        z_c = pixels_in[2, i]
        u = int(pixels_in[0, i] / z_c)
        v = int(pixels_in[1, i] / z_c)
        if is_in_view(u, v, z_c, height, width):
            depth[v,u] = coordinates_cam[2,i] 

    return depth

def depth_to_pointcloud(depth, rgb, K):
    pc = np.empty((0,6), float)
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    height = rgb.shape[0]
    width = rgb.shape[1]

    U = np.tile(np.arange(1, width+1), (height, 1))
    V = np.tile(np.arange(1, height+1), (width, 1)).transpose()
    X = np.multiply(U-cx, depth) / fx
    Y = np.multiply(V-cy, depth) / fy
    for v in range(0, height):
        for u in range(0, width):
            if depth[v,u] == 0:
                continue
            pt = np.array([[X[v,u], Y[v,u], depth[v,u], rgb[v,u,0], rgb[v,u,1], rgb[v,u,2]]])
            pc = np.append(pc, pt, axis=0)
    return pc

def overlay_rgb_depth(rgb, depth):
    height = rgb.shape[0]
    width = rgb.shape[1]
    overlay = np.copy(rgb)
    for v in range(0, height):
        for u in range(0, width):
            if depth[v,u] > 0:
                overlay[v,u,:]=[0,1,1]
    return overlay

def write_hdf5(filename, left, right, pose):
    if os.path.exists(filename):
        os.remove(filename)
    print(filename)
    file = h5py.File(filename, 'w')

    # create datasets
    left_rgb_dataset = file.create_dataset("left_rgb", left[0].shape, 'float16', compression="gzip")
    left_depth_dataset = file.create_dataset("left_depth", left[1].shape, 'float16', compression="gzip")
    right_rgb_dataset = file.create_dataset("right_rgb", right[0].shape, 'float16', compression="gzip")
    right_depth_dataset = file.create_dataset("right_depth", right[1].shape, 'float16', compression="gzip")
    pose_dataset = file.create_dataset("pose", pose.shape, 'float16', compression="gzip")
    
    # write contents into datasets
    left_rgb_dataset[...] = left[0]
    left_depth_dataset[...] = left[1]
    right_rgb_dataset[...] = right[0]
    right_depth_dataset[...] = right[1]
    pose_dataset[...] = pose

    file.close()

def read_hdf5(filename):
    assert os.path.exists(filename)

    file = h5py.File(filename, "r")
    left_rgb = np.array(file['left_rgb']).astype(float)
    left_depth = np.array(file['left_depth']).astype(float)
    right_rgb = np.array(file['right_rgb']).astype(float)
    right_depth = np.array(file['right_depth']).astype(float)
    pose = np.array(file['pose']).astype(float)
    file.close()

    return [left_rgb, left_depth], [right_rgb, right_depth], pose