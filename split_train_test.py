
# import itertools
import matplotlib.pyplot as plt
import numpy as np
import h5py
from mpl_toolkits.mplot3d import Axes3D
import os
from multiprocessing.dummy import Pool as ThreadPool 

import pykitti

__author__ = "Fangchang Ma"
__email__ = "fcma@mit.edu"

do_display = False
do_write = True

# Change this to the directory where you store KITTI data
basedir = '/home/fangchangma/dataset/KITTI_Dataset/dataset'
outDir = '/home/fangchangma/KITTI/'

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

	targetDir = os.path.join(outDir, split, sequence)
	if do_write:
		if not os.path.exists(targetDir):
			os.mkdir(targetDir)

	# Create data iterator
	gray_iterator = dataset.gray
	rgb_iterator = dataset.rgb
	velo_iterator = dataset.velo

	np.set_printoptions(precision=4, suppress=True)
	if do_display:
		f, ax = plt.subplots(3, 2, figsize=(15, 5))
		f2 = plt.figure()

	i = 0
	for first_rgb in rgb_iterator:
		i = i+1
		print('Sequence ' + sequence + ':' + '%05d' % i)

		third_velo = next(velo_iterator)

		# Create depth image
		height = first_rgb[0].shape[0]
		width = first_rgb[0].shape[1]
		# print('height=' + str(height) + ' width=' + str(width))
		depth_image2 = create_depth_image(dataset.calib.P_rect_20, dataset.calib.T_cam2_velo, third_velo, height, width)
		depth_image3 = create_depth_image(dataset.calib.P_rect_30, dataset.calib.T_cam3_velo, third_velo, height, width)

		if do_display:
			# Create overlay image
			overlay2 = overlay_rgb_depth(first_rgb[0], depth_image2)
			overlay3 = overlay_rgb_depth(first_rgb[1], depth_image3)
			
			ax[0, 0].imshow(first_rgb[0])
			ax[0, 1].imshow(first_rgb[1])
			ax[1, 0].imshow(depth_image2, cmap="hot")
			ax[1, 1].imshow(depth_image3, cmap="hot")
			ax[2, 0].imshow(overlay2)
			ax[2, 1].imshow(overlay3)

			# Create point cloud
			pc = depth_to_pointcloud(depth_image2, first_rgb[0], dataset.calib.K_cam2)

			ax2 = f2.add_subplot(111, projection='3d')
			ax2.set_aspect('equal', 'datalim')

			# velo_range = range(0, pc.shape[0], 100)
			N = pc.shape[0]
			velo_range = range(0, N, 2)

			X = pc[velo_range, 0]
			Y = pc[velo_range, 2]
			Z = -pc[velo_range, 1]
			ax2.scatter(X, Y, Z, c=pc[velo_range, 3:6], s=2) 
			ax2.set_title('Colored Point Cloud (subsampled)')
			ax2.set_xlabel('X axis')
			ax2.set_ylabel('Y axis')
			ax2.set_zlabel('Z axis')

			# Create cubic bounding box to simulate equal aspect ratio
			max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
			Xb = 0.1*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
			Yb = 0.1*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
			Zb = 0.1*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
			for xb, yb, zb in zip(Xb, Yb, Zb):
			   ax2.plot([xb], [yb], [zb], 'w')

			# plt.show()
			plt.pause(0.5)

		if do_write:
			filenameL = os.path.join(targetDir, '%05d-L.h5' % i)
			filenameR = os.path.join(targetDir, '%05d-R.h5' % i)
			write_to_hdf5(filenameL, first_rgb[0], depth_image2)
			write_to_hdf5(filenameR, first_rgb[1], depth_image3)

def main():
	trainDir = os.path.join(outDir, 'train')
	testDir = os.path.join(outDir, 'test')

	# Create training set, 00 - 10
	if not os.path.exists(trainDir):
		os.mkdir(trainDir)

	pool = ThreadPool(10) 
	# for i in range(11):
	# 	sequence = '%02d' % i
	# 	iterate_sequence(sequence, 'train')
	sequences = ['%02d' % i for i in range(11)]
	splits = ['train' for i in range(11)]
	pool.starmap(iterate_sequence, zip(sequences, splits))

	# Create test set, 11 - 21
	if not os.path.exists(testDir):
		os.mkdir(testDir)
	# for i in range(11, 22):
	# 	sequence = '%02d' % i
	# 	iterate_sequence(sequence, 'test')
	sequences = ['%02d' % i for i in range(11, 22)]
	splits = ['test' for i in range(11, 22)]
	pool.starmap(iterate_sequence, zip(sequences, splits))

	pool.close() 
	pool.join() 

if __name__ == "__main__":
    main()
