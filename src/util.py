# -*- coding: utf-8 -*-

import os
import sys
import pdb
import yaml

import logging


import pickle
import numpy
import skimage
from skimage import io, transform


import torch

#----------------------------------------------------------------------------Training Device

# For device agnostic 
get_training_device =  lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ['#2DCC69', '#B6BA78', '#4D8E3C', '#F96E2F', '#923177', '#D94DC8',  '#66CDF8', '#8186E2', '#7421F6', '#EC0715', '#98EF89']



#----------------------------------------------------------------------------Dot the Dict

# [How to use a dot “.” to access members of dictionary?] 
# (https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary)
class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


# For nested dictionary. 
class DotDict(dict): 
	"""dot.notation access to dictionary attributes
	   Workds for nested dictionary as well.
	""" 
	def __getattr__(*args): 
		val = dict.get(*args) 
		return DotDict(val) if type(val) is dict else val
	
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__



#----------------------------------------------------------------------------Data & T/V/T Dir Datapath

K_PROJECT_DIR =  os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # os.path.dirname(os.getcwd())
get_src_dir = lambda: os.path.dirname(os.path.abspath(__file__))
get_project_dir = lambda: K_PROJECT_DIR

# pdb.set_trace()


# K_DATA_DIR = os.path.join(K_PROJECT_DIR, 'data')
K_DATA_DIR = os.path.join(os.environ['HOME'], 'data/ada_data')
get_root_data_dir = get_data_dir = lambda data_dir='': os.path.join(K_DATA_DIR, data_dir) # In case there is further depth to directory
get_rel_data_dir = lambda data_dir='': os.path.join('data', data_dir)
get_img_data_dir = lambda img_dir: os.path.join(get_root_data_dir(), img_dir)


get_train_datapath = lambda: os.path.join(get_data_dir(), 'train')
get_val_datapath = lambda: os.path.join(get_data_dir(), 'val')
get_test_datapath = lambda: os.path.join(get_data_dir(), 'test')


get_train_info_datapath = lambda: os.path.join(get_data_dir(), 'train_iminfo.csv')
get_val_info_datapath = lambda: os.path.join(get_data_dir(), 'val_iminfo.csv')
get_test_info_datapath = lambda: os.path.join(get_data_dir(), 'test_iminfo.csv')
get_all_info_datapath = lambda: os.path.join(get_data_dir(), 'all_iminfo.csv')


get_normalization_info_pickle_path = lambda: os.path.join(get_data_dir(), 'normalization_info.pkl')


#---------------------------------------------------------------------------- Models and Results

get_models_dir = lambda: os.path.join(K_PROJECT_DIR, 'models')
get_results_dir = lambda arg='': os.path.join(K_PROJECT_DIR, 'results' , arg)

# ONNX
get_onnx_dir = lambda : os.path.join(K_PROJECT_DIR, 'onnx')


#----------------------------------------------------------------------------CONFIG

config_file = os.path.join(get_src_dir(), 'training_config_init.yaml')
with open(config_file) as f:
	config_text = f.read()


config_yaml = yaml.safe_load(config_text)
kconfig = DotDict(config_yaml)

#----------------------------------------------------------------------------
 
K_TRAINED_MODELNAME = kconfig.model.name
K_MDL_EXT = 'mdl'
def set_trained_model_name(ext_cmt=''):
	global K_TRAINED_MODELNAME
	main_mdl_name = K_TRAINED_MODELNAME.split('.')[0]
	K_TRAINED_MODELNAME = f"{main_mdl_name}{ext_cmt}.{K_MDL_EXT}"
	return K_TRAINED_MODELNAME


get_trained_model_name = lambda: K_TRAINED_MODELNAME
def get_custom_model_name(custom_cmt=''):
	mdl_name = get_trained_model_name()

	main_mdl_name = mdl_name.split('.')[0]

	custom_mdl_name = f'{main_mdl_name}{custom_cmt}.{K_MDL_EXT}'

	return custom_mdl_name


#----------------------------------------------------------------------------Logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
def reset_logger(filename='train_output.log'):
	logger.handlers = []
	filepath = os.path.join(get_results_dir(), filename)
	logger.addHandler(logging.FileHandler(filepath, 'w'))


def add_logger(filename):
	filepath = os.path.join(get_results_dir(), filename)
	logger.addHandler(logging.FileHandler(filepath, 'w'))


def setup_logger(filename='output.log'):
	filepath = os.path.join(get_results_dir(), filename)
	logger.addHandler(logging.FileHandler(filepath, 'a'))
	return logger

reset_logger()


#----------------------------------------------------------------------------Pickle py objects

class PickleHandler(object):    
	@staticmethod
	def dump_in_pickle(py_obj, filepath):
		"""Dumps the python object in pickle
			
			Parameters:
			-----------
				py_obj (object): the python object to be pickled.
				filepath (str): fullpath where object will be saved.
			
			Returns:
			--------
				None
		"""
		with open(filepath, 'wb') as pfile:
			pickle.dump(py_obj, pfile)
	
	
	
	@staticmethod
	def extract_from_pickle(filepath):
		"""Extracts python object from pickle
			
			Parameters:
			-----------
				filepath (str): fullpath where object is pickled
			
			Returns:
			--------
				py_obj (object): python object extracted from pickle
		"""
		with open(filepath, 'rb') as pfile:
			py_obj = pickle.load(pfile)
			return py_obj    			


#----------------------------------------------------------------------------

from collections import Mapping
def pretty(d, indent=0):
	""" Pretty printing of dictionary """
	ret_str = ''
	for key, value in d.items():

		if isinstance(value, Mapping):
			ret_str = ret_str + '\n' + '\t' * indent + str(key) + '\n'
			ret_str = ret_str + pretty(value, indent + 1)
		else:
			ret_str = ret_str + '\n' + '\t' * indent + str(key) + '\t' * (indent + 1) + ' => \n' + str(value) + '\n'

	return ret_str	





#----------------------------------------------------------------------------

if __name__ == '__main__':
	print('Util Module...')





