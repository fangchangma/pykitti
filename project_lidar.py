import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool 
from utils import create_depth_image, overlay_rgb_depth, write_hdf5
import pykitti

__author__ = "Fangchang Ma"
__email__ = "fcma@mit.edu"

do_display = False
do_write = not do_display # implicitly use parallel computing

# Change this to the directory where you store KITTI data
# basedir = '/workspace/KITTI'
basedir = '/home/fangchangma/dataset/KITTI_Dataset/odometry/dataset'
outDir = '/home/fangchangma/dataset/KITTI_Dataset/odometry/'
trainDir = os.path.join(outDir, 'train')
testDir = os.path.join(outDir, 'test')
if not os.path.exists(trainDir):
    os.mkdir(trainDir)
if not os.path.exists(testDir):
    os.mkdir(testDir)

if do_display:
    f, ax = plt.subplots(2, 1, figsize=(12, 12))
    f.tight_layout()

def iterate_sequence(sequence, split):
    # Load the data. Optionally, specify the frame range to load.
    # Passing imformat='cv2' will convert images to uint8 and BGR for
    # easy use with OpenCV.
    # dataset = pykitti.odometry(basedir, sequence, frames=range(0, 500, 500))
    dataset = pykitti.odometry(basedir, sequence)
    print(sequence)

    targetDir = os.path.join(outDir, split, sequence)
    if do_write:
        if not os.path.exists(targetDir):
            os.mkdir(targetDir)

    # pprint(dict(dataset.calib._asdict()))
    K_left, K_right = dataset.calib.K_cam2, dataset.calib.K_cam3

    # Create data iterator
    # gray_iterator = dataset.gray
    rgb_iterator = dataset.rgb
    velo_iterator = dataset.velo

    if split == "train":
        pose_iterator = dataset.poses
    

    i = 0
    for rgbs in rgb_iterator:
        i = i+1
        
        left_rgb = rgbs[0]
        right_rgb = rgbs[1]

        velodyne = next(velo_iterator)
        if split == "train":
            pose = next(pose_iterator)
        else:
            pose = None

        # Create depth image
        height = left_rgb.shape[0]
        width = left_rgb.shape[1]

        left_depth = create_depth_image(K_left, dataset.calib.T_cam2_velo, velodyne, height, width)
        right_depth = create_depth_image(K_right, dataset.calib.T_cam3_velo, velodyne, height, width)

        if do_display:
            # Create overlay image
            left_overlay = overlay_rgb_depth(left_rgb, left_depth)
            # right_overlay = overlay_rgb_depth(right_rgb, right_depth)
            
            ax[0].imshow(left_overlay)
            ax[0].axis("off")
            ax[1].imshow(left_depth>0)
            ax[1].axis("off")
            # ax[2].imshow(depth_image_proj, cmap="hot")
            # ax[2].axis("off")

            print('Sequence ' + sequence + ':' + '%05d' % i + ' rgb.shape=' + str(left_rgb.shape) )

            # plt.show()
            plt.pause(0.5)

        if do_write:
            filename = os.path.join(targetDir, '%05d.h5' % i)
            write_hdf5(filename, [left_rgb, left_depth], [right_rgb, right_depth], pose)

def main():

    if do_write:
        pool = ThreadPool(10) 

        # # Create training set, 00 - 10
        # sequences = ['%02d' % i for i in range(11)]
        # splits = ['train' for i in range(11)]
        # pool.starmap(iterate_sequence, zip(sequences, splits))

        # Create test set, 11 - 21
        sequences = ['%02d' % i for i in range(11, 22)]
        splits = ['test' for i in range(11, 22)]
        pool.starmap(iterate_sequence, zip(sequences, splits))

        pool.close() 
        pool.join() 

    else:
        # Create training set, 00 - 10
        for i in range(11):
            sequence = '%02d' % i
            iterate_sequence(sequence, 'train')
        
        # Create test set, 11 - 21
        for i in range(11, 22):
            sequence = '%02d' % i
            iterate_sequence(sequence, 'test')
    

if __name__ == "__main__":
    main()
