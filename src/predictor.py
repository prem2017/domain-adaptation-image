# -*- coding: utf-8 -*-


# ©Prem Prakash
# Predictor module

import os
import sys
import pdb
from copy import deepcopy
import argparse

import onnx
import onnxruntime as ort

import math
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from .models import ADAConvNet, PretrainedADAConvNet 
from .image_dataset import ImageDataset
from .transformers import NormalizeImageData

from .report_gen_helper import compute_label_and_prob, compute_prob_all, gen_metric_report
from .report_gen_helper import plot_roc_curves_binclass, plot_roc_curves_multiclass

from . import util 
from .util import kconfig
from .util import logger


#----------------------------------------------------------------------------

logger = util.setup_logger('predictor_output.log')
device = util.get_training_device()


#----------------------------------------------------------------------------

def load_trained_model(model_fname, use_batchnorm=False):
	"""Loads the pretrained model for the given model name."""

	model_path = os.path.join(util.get_models_dir(), model_fname)
	model = {}

	# NN Args
	net_args = {}
	net_args['in_channels'] = 3
	net_args['num_cls_lbs'] = kconfig.img.num_cls_lbs
	net_args['num_src_lbs'] = kconfig.img.num_src_lbs
	net_args['model_img_size'] = kconfig.img.size
	net_args['nonlinearity_function'] = None # nn.LeakyReLU()
	net_args['use_batchnorm'] = use_batchnorm # False
	net_args['dropout'] = kconfig.hp.train.dropout 


	if kconfig.tr.use_pretrained_flag:
		# net_args['eps'] = 1e-3
		model = PretrainedADAConvNet(**net_args)
	else:
		model = ADAConvNet(**net_args) 


	saved_state_dict = torch.load(model_path, map_location= lambda storage, loc: storage)

	# Available dict
	# raw_model_dict = model.state_dict()
	

	model.load_state_dict(saved_state_dict)
	model = model.eval()

	return model



#----------------------------------------------------------------------------

# https://pytorch.org/docs/stable/onnx.html
def save_in_onnx(model, onnx_path):
	# First set the model_name and load 

	# pdb.set_trace()
	h, w =  kconfig.img.size
	dummy_input = torch.randn(1, 3, h, w)

	print('[Dummy Output] = ', model(dummy_input))

	input_names = ['image_input']
	output_names = ['output_val']
	# TODO:
	torch.onnx.export(model, dummy_input, onnx_path, verbose=True, 
		input_names=input_names, output_names=output_names, 
		dynamic_axes={'image_input': {0: 'batch'}, 'output_val': {0: 'batch'}})

	print('[Saved] at path = ', onnx_path)



#----------------------------------------------------------------------------
def load_from_onnx(onnx_path):

	pdb.set_trace()
	model_onnx = onnx.load(onnx_path)

	# Check that the IR is well formed
	print(onnx.checker.check_model(model_onnx))

	# Print a human readable representation of the graph
	print(onnx.helper.printable_graph(model_onnx.graph))

	h, w =  kconfig.img.size # util.get_model_img_size()
	dummy_input = np.random.randn(2, 3, h, w).astype(np.float32) # astype(np.float32) is needed to conform to the datatype

	# Load onnx to torch model
	ort_session = ort.InferenceSession(onnx_path)

	outputs = ort_session.run(['output_val'], {'image_input': dummy_input})

	print(outputs)

	return model_onnx



#----------------------------------------------------------------------------

def generate_report(model, test_dataloader, image_names, model_name, save_all_class_prob=False, model_type='test_set'):
	""" all_prob=

	"""
		
	y_true_all = None
	y_pred_all = None

	y_pred_prob_all = None
	y_pred_label_all = None

	# TODO: 
	with torch.no_grad():
		for i, xy in enumerate(test_dataloader):
			print('[Predicting for] batch i = ', i)
			x, y = xy

			y_cls_lbs, y_src_lbs = y
			x = x.to(device=device, dtype=torch.float32) # or float is alias for float32
			
			y_check = y_cls_lbs
			if math.isnan(y_check[0].item()):
				y = None
			else:
				# print(f'[Class Labels Counts] = {y_cls_lbs.unique(return_counts=True)   }', end='')
				# print(f'[Source Labels Counts] = {y_src_lbs.unique(return_counts=True)   }', end='')
				y_cls_lbs = y_cls_lbs.to(device=device, dtype=torch.long)
				y_src_lbs = y_src_lbs.to(device=device, dtype=torch.long)

				y = (y_cls_lbs, y_src_lbs)


			y_pred = model(x)
			y_pred_cls, y_pred_src = y_pred

			# Because we need to only generate 'metric' on class labels i.e. object classes. It is not for the origin of data. 
			if y is not None: # if label is not none
				y_true_all = y_cls_lbs if y_true_all is None else torch.cat((y_true_all, y_cls_lbs), dim=0)	
			y_pred_all = y_pred_cls if y_pred_all is None else torch.cat((y_pred_all, y_pred_cls), dim=0)

			
				
	
	num_classes = kconfig.img.num_cls_lbs # len(util.get_class_names())
	if y_true_all is not None:
		report, f1_checker, auc_val = gen_metric_report(y_true_all, y_pred_all)
		if num_classes > 2:
			plot_roc_curves_multiclass(report, 'inference_time', model_type= model_type, ext_cmt=model_name)
		else:
			plot_roc_curves_binclass(report, 'inference_time', model_type= model_type, ext_cmt=model_name)

		report['roc'] = 'Removed'
		msg = util.pretty(report)
		logger.info(msg); print(msg)
		y_true_all = y_true_all.cpu().numpy().reshape(-1, 1)


	y_pred_max_prob_all, y_pred_label_all = compute_label_and_prob(y_pred_all)
	y_pred_label_all = y_pred_label_all.cpu().numpy().reshape(-1, 1)
	y_pred_max_prob_all = y_pred_max_prob_all.cpu().numpy().reshape(-1, 1)


	round_upto = 4 
	y_pred_max_prob_all = (y_pred_max_prob_all * 10**round_upto).astype(int) / 10**4 


	class_names = kconfig.img.labels_dict.values() 
	# output prediction in csv
	if image_names is not None:
		
		image_names = image_names.reshape(-1, 1)

		if save_all_class_prob and num_classes > 2:
			header = ['image_name'] + [cl_name + '_class_prob' for cl_name in class_names] + ['pred_label']
			y_pred_prob_all = compute_prob_all(y_pred_all)
			y_pred_prob_all = y_pred_prob_all.cpu().numpy()
			y_pred_prob_all = (y_pred_prob_all * 10**round_upto).astype(int) / 10**round_upto
			df = np.hstack((image_names, y_pred_prob_all, y_pred_label_all))
		else:
			header = ['image_name', 'pred_label_prob', 'pred_label']
			df = np.hstack((image_names, y_pred_max_prob_all, y_pred_label_all))

		if y_true_all is not None:
			header.append('true_label')
			df = np.hstack((df, y_true_all))
		
		df = pd.DataFrame(df, columns=header)
		df.to_csv(path_or_buf=os.path.join(util.get_results_dir(), model_name + '_test_output_prediction.csv'), sep=',', index=None, header=header)

	return 



#----------------------------------------------------------------------------

# TODO: use model to get an prediction from command line
def get_arguments_parser(img_datapath):
	"""Argument parser for predition"""
	description = 	'Provide arguments for fullpath to an images to receive prob score and class lable.'
					
	parser = argparse.ArgumentParser(description=description)


	parser.add_argument('-i', '--img', type=str, default=img_datapath, 
		help='Provide full path to image location.', required=True)


	return parser




#----------------------------------------------------------------------------

def main_onnx(also_test=False):
	# pdb.set_trace()

	util.set_trained_model_name(ext_cmt='scratch') 
	base_model_fname = util.get_trained_model_name()
	use_batchnorm = kconfig.tr.use_batchnorm

	cmt = '' # '_maxf1'
	model_fname = base_model_fname + cmt
	model = load_trained_model(model_fname, use_batchnorm)
	
	onnx_path = os.path.join(util.get_onnx_dir(), model_fname + '.onnx')

	save_in_onnx(model, onnx_path)
	print('[Saved] in ONNX format')

	if also_test:
		load_from_onnx(onnx_path)




#----------------------------------------------------------------------------

def main():

	util.reset_logger('predictor_output.log')

	# First set the model_name and load 
	util.set_trained_model_name(ext_cmt='_plain') #  ['scratch', 'pretrained_resnet50']
	base_model_fname = util.get_trained_model_name()


	use_batchnorm = kconfig.tr.use_batchnorm
	img_info_datapath = util.get_test_info_datapath() # get_all_info_datapath() # get_test_info_datapath


	# Dataset Args 
	data_dir =  kconfig.img.data_dir # ''

	data_info_args = {}
	data_info_args['root_data_dir'] = util.get_data_dir(data_dir)
	data_info_args['model_img_size'] = kconfig.img.size
	data_info_args['normalizer_dict_path'] = util.get_normalization_info_pickle_path()
	data_info_args['has_label'] = True
	data_info = {'img_info_datapath': img_info_datapath, **data_info_args}

	test_dataset = ImageDataset(**data_info)
	image_names = test_dataset.get_img_names()
	test_dataloader = DataLoader(dataset=test_dataset, batch_size=kconfig.test.batch_size)


	ex =  '' # 
	# model_fnames = [base_model_fname + ex for ex in ['', '_maxauc', '_maxf1', '_minval', '_mintrain']] #  ['', '_maxauc', '_maxf1', '_minval', '_mintrain'] 
	model_fnames = [util.get_custom_model_name(ex) for ex in ['', '_maxauc', '_maxf1', '_minval', '_mintrain']] #  ['', '_maxauc', '_maxf1', '_minval', '_mintrain'] 


	models = [load_trained_model(model_fname, use_batchnorm) for model_fname in model_fnames]
	output = {}
	for i, model in enumerate(models):
		print(f'\n\n[Generating Report] for model = {model_fnames[i]}\n')
		
		generate_report(model, test_dataloader, image_names=image_names, model_name=model_fnames[i], save_all_class_prob=True)
		
		
	
	print('\n\n######################################################\n')
	print(output)
	print('\n######################################################\n\n')
	return output



#----------------------------------------------------------------------------

if __name__ == '__main__':
	print('[Run Test]')
	main()
	# main_onnx(True)



