import os
import numpy as np
import h5py
from pprint import pprint

def is_in_view(u, v, z_c, height, width):
    return (z_c>0 and u>=0 and u<width and v>=0 and v<height)

def create_depth_image(projection_matrix, intrinsics, T_velo_to_cam, velo, height, width):
    coordinates_cam = np.dot(T_velo_to_cam, velo.transpose())
    N = coordinates_cam.shape[1]

    pixels = np.dot(intrinsics, coordinates_cam[:3,:])
    depth_image_in = np.zeros((height, width))
    for i in range(0, N):
        z_c = pixels[2, i]
        u = int(pixels[0, i] / z_c)
        v = int(pixels[1, i] / z_c)
        if is_in_view(u, v, z_c, height, width):
            depth_image_in[v,u] = coordinates_cam[2,i] 

    pixels = np.dot(projection_matrix, coordinates_cam)[:3, :]
    depth_image_proj = np.zeros((height, width))
    for i in range(0, N):
        z_c = pixels[2, i]
        u = int(pixels[0, i] / z_c)
        v = int(pixels[1, i] / z_c)
        if is_in_view(u, v, z_c, height, width):
            depth_image_proj[v,u] = coordinates_cam[2,i] 

    return depth_image_in, depth_image_proj

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
                overlay[v,u,:]=[0,0,1]
    return overlay

def write_to_hdf5(filename, rgb, depth):
    file = h5py.File(filename, 'w')
    rgb_dataset = file.create_dataset("rgb", rgb.shape, 'uint8', compression="gzip")
    depth_dataset = file.create_dataset("depth", depth.shape, 'float16', compression="gzip")
    rgb_dataset[...] = rgb
    depth_dataset[...] = depth
    file.close()