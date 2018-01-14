import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils import read_hdf5, overlay_rgb_depth

outDir = '/home/fangchangma/dataset/KITTI_Dataset/odometry/'
trainDir = os.path.join(outDir, 'train')
testDir = os.path.join(outDir, 'test')

f, ax = plt.subplots(3, 1, figsize=(12, 12))
f.tight_layout()    

def main():
    filename = os.path.join(trainDir, "03", "00002.h5")
    left, right, pose = read_hdf5(filename)
    # print(left[0].shape, pose.shape)
    print("rgb: max={}, min={}".format(
        np.max(left[0]),  np.min(left[0]),
        ))
    print("depth: max={:.3f}, min={:.3f}".format(
        np.max(left[1]),  np.min(left[1]),
        ))
    print(pose)

    # Create overlay image
    left_overlay = overlay_rgb_depth(left[0], left[1])
    # right_overlay = overlay_rgb_depth(right_rgb, right_depth)
    
    ax[0].imshow(left_overlay)
    ax[0].axis("off")
    ax[1].imshow(left[0])
    ax[1].axis("off")
    ax[2].imshow(left[1], cmap="hot")
    ax[2].axis("off")

    plt.show()
    
if __name__ == "__main__":
    main()