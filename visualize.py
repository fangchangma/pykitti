import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing.dummy import Pool as ThreadPool 
from pprint import pprint
import pykitti
from utils import create_depth_image, overlay_rgb_depth

__author__ = "Fangchang Ma"
__email__ = "fcma@mit.edu"

# Change this to the directory where you store KITTI data
basedir = '/home/fangchangma/dataset/KITTI_Dataset/odometry/dataset'

f, ax = plt.subplots(2, 1, figsize=(12, 12))
f.tight_layout()
def iterate_sequence(sequence, split):

    # dataset = pykitti.odometry(basedir, sequence, frames=range(0, 500, 500))
    dataset = pykitti.odometry(basedir, sequence)

    # pprint(dict(dataset.calib._asdict()))
    K_left, K_right = dataset.calib.K_cam2, dataset.calib.K_cam3

    # Create data iterator
    gray_iterator = dataset.gray
    rgb_iterator = dataset.rgb
    velo_iterator = dataset.velo
    pose_iterator = dataset.poses

    # np.set_printoptions(precision=4, suppress=True)

    i = 0
    for rgbs in rgb_iterator:
        i = i+1
        
        left_rgb = rgbs[0]
        right_rgb = rgbs[1]

        velodyne = next(velo_iterator)
        pose = next(pose_iterator)

        # Create depth image
        height = left_rgb.shape[0]
        width = left_rgb.shape[1]

        print()
        print(pose)

        left_depth = create_depth_image(K_left, dataset.calib.T_cam2_velo, velodyne, height, width)
        # right_depth = create_depth_image(K_right, dataset.calib.T_cam3_velo, velodyne, height, width)

        # Create overlay image
        left_overlay = overlay_rgb_depth(left_rgb, left_depth)
        # right_overlay = overlay_rgb_depth(right_rgb, right_depth)
        
        ax[0].imshow(left_overlay)
        ax[0].axis("off")
        ax[1].imshow(left_depth>0)
        ax[1].axis("off")
        # ax[2].imshow(depth_image_proj, cmap="hot")
        # ax[2].axis("off")

        print('Sequence ' + sequence + ':' + '%05d' % i + ' rgb.shape=' + str(left_rgb.shape) + ' cropped shape=' + str(left_rgb[130:370, :, :].shape))

        # plt.show()
        plt.pause(0.5)

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
