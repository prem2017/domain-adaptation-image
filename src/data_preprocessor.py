# -*- coding: utf-8 -*-

import sys
import os
import random
import pdb
import glob

import numpy as np
import pandas as pd

from skimage import io
from skimage.color import rgba2rgb


from . import util
from .util import kconfig
from .transformers import resize_image





#----------------------------------------------------------------------------Train/Val- iminfo generation 

def save_multi_sources_train_val_imgs(img_dirpath, train_sources_dict, labels_dict, train_frac=0.9):

	# Structure
	#{'source_origin': {'class_name': img_name}}

	columns = ['img_relpath', 'class_label', 'source_label']
	train_info = []
	val_info = []

	# pdb.set_trace()
	for src_id, source in train_sources_dict.items():
		full_img_dirpath_src = os.path.join(img_dirpath, source)

		store_img_relpath_src = os.path.join(source)

		for lbl_id, label in labels_dict.items():

			full_img_dirpath_label = os.path.join(full_img_dirpath_src, label)

			store_img_relpath_label = os.path.join(store_img_relpath_src, label)

			# Load all the images
			imgs_relpath = [os.path.join(store_img_relpath_label, img) for img in os.listdir(full_img_dirpath_label) if not img.startswith('.')]

			random.shuffle(imgs_relpath)

			train_len = int(len(imgs_relpath) * train_frac)

			src_lb_train_data = [[img_relpath, lbl_id, src_id] for img_relpath in imgs_relpath[:train_len]]
			src_lb_val_data = [[img_relpath, lbl_id, src_id] for img_relpath in imgs_relpath[train_len:]]

			train_info += src_lb_train_data
			val_info += src_lb_val_data


	# pdb.set_trace()
	random.shuffle(train_info)
	df_train_info = pd.DataFrame(data=train_info, columns=columns)
	df_val_info = pd.DataFrame(data=val_info, columns=columns)

	df_train_info.to_csv(util.get_train_info_datapath(), index=False)
	df_val_info.to_csv(util.get_val_info_datapath(), index=False)

	return df_train_info, df_val_info

#----------------------------------------------------------------------------Test- iminfo generation

def save_multi_sources_test_imgs(img_dirpath, test_sources_dict, labels_dict):

	# Structure
	#{'source_origin': {'class_name': img_name}}

	columns = ['img_relpath', 'class_label', 'source_label']
	test_info = []


	# pdb.set_trace()
	for src_id, source in test_sources_dict.items():
		full_img_dirpath_src = os.path.join(img_dirpath, source)

		store_img_relpath_src = os.path.join(source)

		for lbl_id, label in labels_dict.items():

			full_img_dirpath_label = os.path.join(full_img_dirpath_src, label)

			store_img_relpath_label = os.path.join(store_img_relpath_src, label)

			# Load all the images
			imgs_relpath = [os.path.join(store_img_relpath_label, img) for img in os.listdir(full_img_dirpath_label) if not img.startswith('.')]

			imgs_relpath = [[img_relpath, lbl_id, src_id] for img_relpath in imgs_relpath]
			test_info += imgs_relpath



	# pdb.set_trace()
	df_test_info = pd.DataFrame(data=test_info, columns=columns)

	df_test_info.to_csv(util.get_test_info_datapath(), index=False)


	return




#----------------------------------------------------------------------------Normalization Computation


def compute_img_normalization_vals(df_train_info, img_dirpath, normalization_vals_dictpath):


	img_len = df_train_info.shape[0]

	imgs = [None] * img_len 

	# pdb.set_trace()
	for i, row in df_train_info.iterrows():

		# if i == 20:
		# 	break

		img_fullpath = os.path.join(img_dirpath, row.img_relpath)

		img = io.imread(img_fullpath)
		
		img = resize_image(img, kconfig.img.size)

		print(i, ': ', row.img_relpath, ', ', img.shape)

		img = img / 255.0 if img.shape[2] == 3 else rgba2rgb(img) # shape = HxWxC # TODO: take care if is gray-scale image

		imgs[i] = img


	# pdb.set_trace()
	# Change datatype for easy computation
	imgs = np.array(imgs)
	print(imgs.shape)
	imgs_mean = np.round(imgs.mean(axis=(0, 1, 2)), decimals=3)
	imgs_std = np.round(imgs.std(axis=(0, 1, 2)), decimals=3)

	normalization_vals_dict = {'mean': imgs_mean.tolist(), 'std': imgs_std.tolist()}
	print('[Normalization] vals = ', normalization_vals_dict)

	util.PickleHandler.dump_in_pickle(normalization_vals_dict, normalization_vals_dictpath)

	return normalization_vals_dict




#----------------------------------------------------------------------------

if __name__ == '__main__':
	print('Preprocessor')

	

	img_dir = 'shts_to_yoga' # 'tbl_to_wngl'
	img_dirpath = util.get_img_data_dir(img_dir)

	# pdb.set_trace()

	train_sources_dict = kconfig.img.train_sources_dict # ['quickdraw'. 'real'. 'sketch'] 
	test_sources_dict = kconfig.img.test_sources_dict # ['infograph'] 

	labels_dict = kconfig.img.labels_dict

	

	# Data Info
	# source_type/class/image_name.ext,class_label,source_label
	df_train_info, _ = save_multi_sources_train_val_imgs(img_dirpath, train_sources_dict, labels_dict)
	save_multi_sources_test_imgs(img_dirpath, test_sources_dict, labels_dict)


	# Compute normalizations
	normalization_vals_dictpath = util.get_normalization_info_pickle_path()

	compute_img_normalization_vals(df_train_info, img_dirpath, normalization_vals_dictpath)



	print('************* Completed *************')








