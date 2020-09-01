# -*- coding: utf-8 -*- 

import os
import random
import pdb

import math
import numpy as np
import pandas as pd

from skimage import io
from skimage.color import rgba2rgb

import torch
from torch.utils.data import Dataset, DataLoader


from . import util
from .transformers import resize_image, load_and_resize_image, NormalizeImageData


# util.PickleHandler.extract_from_pickle(util.get_normalization_info_pickle_path())
# This dataset return tiplet with image_nd_array, class_label, source_label
class ImageDataset(Dataset):
	""" Image dataset for custom reading of images 
		model_img_size = (height, width)
	"""
	
	def __init__(self, root_data_dir, img_info_datapath, transformers=None, model_img_size=(64, 64), normalizer_dict_path=None, has_label=True):
		""" """
		self.root_data_dir = root_data_dir

		self.frows = pd.read_csv(img_info_datapath, delimiter=',', skipinitialspace=True, comment='#')

		# if 'val' in img_info_datapath.lower():
		#	# pdb.set_trace()
		#	self.frows = self.frows.sample(frac=1).reset_index(drop=True)

		
		self.transformers = transformers

		self.model_img_size = model_img_size # (height, width)

		self.normalizer = None
		if normalizer_dict_path is not None:
			normalizer_dict = util.PickleHandler.extract_from_pickle(normalizer_dict_path)
			self.normalizer = NormalizeImageData(**normalizer_dict)


		# pdb.set_trace()
		self.has_label = has_label
	

	def __len__(self):
		return self.frows.shape[0] 


	def get_img_names(self):
		return self.frows.values[:, 0] # os.path.basename(os.path.normpath('a/b/c/abc.jpg')) => abc.jpg

	

	def __getitem__(self, idx):
		
		img_relpath = self.frows.iloc[idx, 0]
		img_fullpath = os.path.join(self.root_data_dir, img_relpath)
		
		img_ndarray = io.imread(img_fullpath) # load_and_resize_image(img_fullpath) 
		
		# Handle the channels of the image
		if len(img_ndarray.shape) == 3:
			if img_ndarray.shape[2] == 4:
				img_ndarray = rgba2rgb(img_ndarray)
		else:  # If b&w image repeat the dimension
			img_ndarray = np.expand_dims(img_ndarray, axis=2)
			img_ndarray = np.concatenate((img_ndarray, img_ndarray, img_ndarray), axis=2)

		# pdb.set_trace()
		if self.model_img_size is not None and self.model_img_size != (img_ndarray.shape[0], img_ndarray.shape[1]):
			img_ndarray = resize_image(img_ndarray, self.model_img_size)

		# Between 0-1
		img_ndarray = img_ndarray / 255

		# pdb.set_trace()
		if self.transformers is not None and len(self.transformers) > 0:
			pick = random.randint(0, max(0, len(self.transformers)-1))
			# print(f'picked[t] = {pick}')
			transformer = self.transformers[pick]
			img_ndarray = transformer(img_ndarray)

		# pdb.set_trace()
		# Normalize the data
		if self.normalizer is not None:
			img_ndarray = self.normalizer(img_ndarray)


		img_ndarray = img_ndarray.transpose(2, 0, 1)


		# pdb.set_trace()

		if self.has_label:
			class_label, source_label = self.frows.iloc[idx, 1:]
			return img_ndarray, (class_label, source_label)
		else:
			return img_ndarray, (np.nan, np.nan) # np.nan is to represent the None type that is- class-label and source-label is not available




