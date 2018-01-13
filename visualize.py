
# import itertools
import matplotlib.pyplot as plt
import numpy as np
import h5py
from mpl_toolkits.mplot3d import Axes3D
import os
from multiprocessing.dummy import Pool as ThreadPool 
from pprint import pprint

import pykitti

__author__ = "Fangchang Ma"
__email__ = "fcma@mit.edu"

do_display = True

if do_display:
    f, ax = plt.subplots(3, 1, figsize=(12, 12))
    f.tight_layout()
    # f2 = plt.figure()

# Change this to the directory where you store KITTI data
basedir = '/home/fangchangma/dataset/KITTI_Dataset/odometry/dataset'

def is_in_view(u, v, z_c, height, width):
    return (z_c>0 and u>=0 and u<width and v>=0 and v<height)

def create_depth_image(Pi, Tr, velo, height, width):
    coordinates_cam = np.dot(Tr, velo.transpose())
    pixels = np.dot(Pi, coordinates_cam)
    N = pixels.shape[1]
    depth_image = np.zeros((height, width))
    for i in range(0, N):
        z_c = pixels[2, i]
        u = int(pixels[0, i] / z_c)
        v = int(pixels[1, i] / z_c)
        if is_in_view(u, v, z_c, height, width):
            depth_image[v,u] = coordinates_cam[2,i] 
    return depth_image

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

def iterate_sequence(sequence, split):
    # Load the data. Optionally, specify the frame range to load.
    # Passing imformat='cv2' will convert images to uint8 and BGR for
    # easy use with OpenCV.
    # dataset = pykitti.odometry(basedir, sequence, frames=range(0, 500, 500))
    dataset = pykitti.odometry(basedir, sequence)

    # Create data iterator
    gray_iterator = dataset.gray
    rgb_iterator = dataset.rgb
    velo_iterator = dataset.velo
    pose_iterator = dataset.poses

    np.set_printoptions(precision=4, suppress=True)

    i = 0
    for rgbs in rgb_iterator:
        i = i+1
        
        left_rgb = rgbs[0]
        right_rgb = rgbs[1]
        

        # if (i>1):
        #   break

        third_velo = next(velo_iterator)
        pose = next(pose_iterator)

        if i < 15:
            continue

        # Create depth image
        height = left_rgb.shape[0]
        width = left_rgb.shape[1]
        # print('height=' + str(height) + ' width=' + str(width))
        print()
        pprint(dict(dataset.calib._asdict()))
        # print(dataset.calib.P_rect_20)
        # print(dataset.calib.P_rect_30)
        print(pose)
        left_depth = create_depth_image(dataset.calib.P_rect_20, dataset.calib.T_cam2_velo, third_velo, height, width)
        right_depth = create_depth_image(dataset.calib.P_rect_30, dataset.calib.T_cam3_velo, third_velo, height, width)

        if do_display:
            # Create overlay image
            left_overlay = overlay_rgb_depth(left_rgb, left_depth)
            right_overlay = overlay_rgb_depth(right_rgb, right_depth)
            
            ax[0].imshow(left_rgb)
            ax[0].axis("off")
            ax[1].imshow(left_depth, cmap="hot")
            ax[1].axis("off")
            ax[2].imshow(left_overlay)
            ax[2].axis("off")


            # ax[0, 0].imshow(left_rgb)
            # ax[0, 0].axis("off")

            # # ax[0, 1].imshow(left_rgb[130:370, :, :])
            # ax[0, 1].imshow(left_rgb)
            # ax[0, 1].axis("off")

            # ax[1, 0].imshow(left_depth, cmap="hot")
            # ax[1, 0].axis("off")

            # # ax[1, 1].imshow(right_depth[130:370, :], cmap="hot")
            # ax[1, 1].imshow(right_depth, cmap="hot")
            # ax[1, 1].axis("off")

            # ax[2, 0].imshow(left_overlay)
            # ax[2, 0].axis("off")

            # # ax[2, 1].imshow(left_overlay[130:370, :, :])
            # ax[2, 1].imshow(left_overlay)
            # ax[2, 1].axis("off")


            print('Sequence ' + sequence + ':' + '%05d' % i + ' rgb.shape=' + str(left_rgb.shape) + ' cropped shape=' + str(left_rgb[130:370, :, :].shape))

            # Create point cloud
            # pc = depth_to_pointcloud(left_depth, rgbs[0], dataset.calib.K_cam2)

            # ax2 = f2.add_subplot(111, projection='3d')
            # ax2.set_aspect('equal', 'datalim')

            # # velo_range = range(0, pc.shape[0], 100)
            # N = pc.shape[0]
            # velo_range = range(0, N, 2)

            # X = pc[velo_range, 0]
            # Y = pc[velo_range, 2]
            # Z = -pc[velo_range, 1]
            # ax2.scatter(X, Y, Z, c=pc[velo_range, 3:6], s=2) 
            # ax2.set_title('Colored Point Cloud (subsampled)')
            # ax2.set_xlabel('X axis')
            # ax2.set_ylabel('Y axis')
            # ax2.set_zlabel('Z axis')

            # # Create cubic bounding box to simulate equal aspect ratio
            # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
            # Xb = 0.1*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
            # Yb = 0.1*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
            # Zb = 0.1*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
            # for xb, yb, zb in zip(Xb, Yb, Zb):
            #    ax2.plot([xb], [yb], [zb], 'w')

            plt.show()
            # plt.pause(0.5)

def main():
    for i in range(11):
      sequence = '%02d' % i
      iterate_sequence(sequence, 'train')

    # # test set, 11 - 21
    # for i in range(11, 22):
    #     sequence = '%02d' % i
    #     iterate_sequence(sequence, 'test')

if __name__ == "__main__":
    main()
