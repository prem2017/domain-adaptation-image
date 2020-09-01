	# -*- coding: utf-8 -*- 


# ©Prem Prakash
# Main module to train the model


import os
import math
import pdb
import time
from datetime import datetime
from itertools import cycle
import argparse


import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


from .models import ADAConvNet, ADAConvNetClsLabel, ADAConvNetSrcLabel, PretrainedADAConvNet 
from .image_dataset import ImageDataset
from .loss_functions import ADALoss, ADAClsLabelLoss, ADASrcLabelLoss 

from .report_gen_helper import gen_metric_report
from .report_gen_helper import plot_roc_curves_binclass, plot_roc_curves_multiclass


from .transformers import RandomNoiseImage, MirrorImage, InvertVerticallyImage
from .transformers import MirrorAndInvertVerticallyImage, Rotate90Image, Rotate270Image 
from .transformers import ChangeHSLImage, ChangeSaturationImage, ChangeLuminescenceImage, ChangeHueImage 

from .import util
from .util import kconfig
from .util import logger



#----------------------------------------------------------------------------

device = util.get_training_device()


#----------------------------------------------------------------------------

class Optimizer(object):
	"""Different optimizer of optimize learning process than vanilla greadient descent """
	def __init__(self):
		super(Optimizer, self).__init__()
		
		
	@staticmethod
	def rmsprop_optimizer(params, lr=1e-3, weight_decay=1e-6):
		return optim.RMSprop(params=params, lr=lr, alpha=0.99, eps=1e-6, centered=True, weight_decay=weight_decay)


	@staticmethod
	def adam_optimizer(params, lr=1e-3, weight_decay=1e-6):
		return optim.Adam(params=params, lr=lr, weight_decay=weight_decay)

	@staticmethod
	def sgd_optimizer(params, lr=1e-6, weight_decay=1e-6, momentum=0.9):
		return optim.SGD(params=params, lr=lr, weight_decay=weight_decay, momentum=momentum)


#----------------------------------------------------------------------------

# TODO:
def print_wt_n_output(training_model, y, output, print_all_weights=False, optimizer=None):
	# TODO: might change in future [Weight] print weights
	# pdb.set_trace()
	layer = training_model.fc2
	fc_wt = layer.weight.data[0, 0:5]
	fc_wt_grad = layer.weight.grad.data[0, 0:5]

	msg = '\nModel FC_wt = {}  \nModel FC_grad = {} '.format(fc_wt, fc_wt_grad)
	print(msg)  # logger.info(msg);
	
	left, right = 10, 20
	y_sample = y[left:right, :]
	output_sample = output.data[left:right, :]
	msg = f'\ny[{left}:{right}, :] = {y_sample}   \noutput.data[{left}:{right}, :] = {output_sample}'
	print(msg)
	
	if print_all_weights:
		tm_state_dict = training_model.cpu().float().state_dict()
		print('\n\n\n[All the weights]')
		for k, v in tm_state_dict.items():
			print('\n\n', k)
			v = np.array(v)
			print(v.shape)
			print(v)
			

	# pitr = iter(optimizer.param_groups[0]['params'])
	# v = next(pitr)
	# from collections import deque
	# dd = deque(pitr, maxlen=1)
	# v = dd.pop()
	if optimizer is not None:
		print('\n\n\n [All the gradient of weights]')
		# optimizer_state_dict = optimizer.state_dict()
		
		# for v in optimizer.param_groups[0]['params']:
		v =  optimizer.param_groups[0]['params'][-2]
		print('\n\n')
		state = optimizer.state[v]
		print('\n\n\n[Gradient] GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG')
		grad = v.grad.data.cpu().numpy()
		print(grad.shape); print(grad[:, left:right]) # TODO: should change depending on model
		for key, key_val in state.items():
			print(f'\n\n[Key] = {key}')
			key_val = key_val.cpu().numpy()
			print(key_val.shape); print(key_val[:, left:right]) # TODO: should change depending on model



#----------------------------------------------------------------------------

def train_network(dataloader, model, loss_function, optimizer, start_lr, end_lr, num_epochs=90, sanity_check=False):
	"""Trains the network and saves for different checkpoints such as minimum train/val loss, f1-score, AUC etc. different performance metrics

		Parameters:
		-----------
			dataloader (dict): {key (str):  Value(torch.utils.data.DataLoader)} training and validation dataloader to respective purposes
			model (nn.Module): models to traine the face-recognition
			loss_function (torch.nn.Module): Module to mesure loss between target and model-output
			optimizer (Optimizer): Non vanilla gradient descent method to optimize learning and descent direction
			start_lr (float): For one cycle training the start learning rate
			end_lr (float): the end learning must be greater than start learning rate
			num_epochs (int): number of epochs the one cycle is 
			sanity_check (bool): if the training is perfomed to check the sanity of the model. i.e. to anaswer 'is model is able to overfit for small amount of data?'

		Returns:
		--------
			None: perfoms the required task of training

	"""

	if isinstance(model, dict):
		for k, v in model.items():
			model[k] = v.train()
	else:
		model = model.train()

	logger_msg = '\nDataLoader = {}' \
				 '\nModel = {}' \
				 '\nLossFucntion = {}' \
				 '\nOptimizer = {}' \
				 '\nStartLR = {}, EndLR = {}' \
				 '\nNumEpochs = {}'.format(dataloader, model, loss_function, optimizer, start_lr, end_lr, num_epochs)

	logger.info(logger_msg), print(logger_msg)

	# [https://arxiv.org/abs/1803.09820]
	# This is used to find optimal learning-rate which can be used in one-cycle training policy
	# [LR]TODO: for finding optimal learning rate

	lr_scheduler = {}
	if kconfig.tr.lr_search_flag:
		if isinstance(optimizer, dict):
			for k, opt in optimizer.items():
				lr_scheduler[k] = MultiStepLR(optimizer=opt, milestones=list(np.arange(2, 24, 2)), gamma=10, last_epoch=-1)
		else:
			lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=list(np.arange(2, 24, 2)), gamma=10, last_epoch=-1)
		

	# TODO: Cyclic momentum
	# optimizer.param_groups[0]['momentum'] # weight_decay
	# 0.95 -> 0.8.
	# This implies that as LR increases during 1Cycle, WD should decrease. 
	# https://forums.fast.ai/t/one-cycle-policy/25944/2
	# The large batch training literature recommends not using WD on BN, so if you are asking what your should do, don’t apply WD to BN.


	def get_lr():
		lr = []
		# pdb.set_trace()

		if isinstance(optimizer, dict): 
			for k, opt in optimizer.items():
				for param_group in opt.param_groups:
					lr.append(np.round(param_group['lr'], 11))
				break
		else:
			for param_group in optimizer.param_groups:
				lr.append(np.round(param_group['lr'], 11))
		return lr



	def set_lr(lr):
		if isinstance(optimizer, dict):
			for k, opt in optimizer.items():
				for param_group in opt.param_groups:
					param_group['lr'] = lr
		else:
			for param_group in opt.param_groups:
				param_group['lr'] = lr


	def set_momentum(m):
		# pdb.set_trace()

		if isinstance(optimizer, dict):
			for k, opt in optimizer.items():
				for param_group in opt.param_groups:
					param_group['momentum'] = m
		else:
			for param_group in opt.param_groups:
					param_group['momentum'] = m


	# 'Training': loss Containers
	train_cur_epoch_batchwise_loss = []
	train_epoch_avg_loss_container = []  # Stores loss for each epoch averged over batches.
	train_all_epoch_batchwise_loss = []

	# 'Validation': loss containers
	val_avg_loss_container = []


	# 'Validation': Metric Containers 
	val_report_container = []
	val_f1_container = []
	val_auc_container = []
	val_accuracy_container = []

	# 'Test': Metric Containers. Only computed and stored when certain condition is met. 
	# Of course, this is perfomed only when test_set with labels are present. 
	test_auc_container = {}
	test_f1_container = {}
	test_accuracy_container = {}

	
	# 'Extra' epochs
	if kconfig.tr.lr_search_flag:
		extra_epochs = kconfig.one_cycle_policy.extra_epochs.lr_search # 4 
	else:
		extra_epochs = kconfig.one_cycle_policy.extra_epochs.train  # 20
	total_epochs = num_epochs + extra_epochs


	# One cycle setting of Learning Rate
	num_steps_upndown = kconfig.one_cycle_policy.num_steps_upndown # 10
	further_lowering_factor = kconfig.one_cycle_policy.extra_epochs.lowering_factor  # 10
	further_lowering_factor_steps = kconfig.one_cycle_policy.extra_epochs.lower_after # 4


	# Cyclic Learning Rate
	def one_cycle_lr_setter(current_epoch):
		start_momentum = 0.95
		end_momentum = 0.85
		current_momentum = None
		if current_epoch <= num_epochs:
			assert end_lr > start_lr, '[EndLR] should be greater than [StartLR]'
			lr_inc_rate = np.round((end_lr - start_lr) / (num_steps_upndown), 9)
			lr_inc_epoch_step_len = max(num_epochs / (2 * num_steps_upndown), 1)

			steps_completed = current_epoch / lr_inc_epoch_step_len
			print('[Steps Completed] = ', steps_completed)
			if steps_completed <= num_steps_upndown:
				current_lr = start_lr + (steps_completed * lr_inc_rate)
				current_momentum = start_momentum - ((start_momentum - end_momentum) * int(steps_completed) / num_steps_upndown)
			else:
				current_lr = end_lr - ((steps_completed - num_steps_upndown) * lr_inc_rate)
				current_momentum = end_momentum + ((start_momentum - end_momentum) * int(steps_completed - num_steps_upndown) / num_steps_upndown)

			set_lr(current_lr)
			# set_momentum(current_momentum)
		else:
			current_lr = start_lr / (
						further_lowering_factor ** ((current_epoch - num_epochs) // further_lowering_factor_steps))
			set_lr(current_lr)



	if sanity_check:
		train_dataloader = next(iter(dataloader['train']))
		train_dataloader = [train_dataloader] * 128
	else:
		train_dataloader = dataloader['train']


	def reset_grad(optimizer):
		# Zero Grad
		if isinstance(optimizer, dict):
			for k, opt in optimizer.items():
				opt.zero_grad()
		else:
			optimizer.zero_grad()


	# Model Tranining for 'total_epochs' 
	counter = 0
	ep_ctr = 0
	for epoch in range(total_epochs):
		msg = '\n\n\n[Epoch] = {}'.format(epoch + 1)
		print(msg)
		start_time = time.time()
		start_datetime = datetime.now()
		
		for i, (X, y) in enumerate(train_dataloader): 

			y_cls_lbs, y_src_lbs = y
			# print(f'[Class Labels Counts] = {y_cls_lbs.unique(return_counts=True)   }', end='')
			# print(f'[Source Labels Counts] = {y_src_lbs.unique(return_counts=True)   }', end='')


			X = X.to(device=device, dtype=torch.float32) # or float is alias for float32

			# pdb.set_trace()


			# ep_ctr += 1
			# if ep_ctr % 2 == 0 and (kconfig.tr.train_flag or kconfig.tr.sanity_check_flag):
			# 	src_idx_randperm = torch.randperm(len(y_src_lbs))
			# 	y_src_lbs = y_src_lbs[src_idx_randperm]


			y_cls_lbs = y_cls_lbs.to(device=device, dtype=torch.long)
			y_src_lbs = y_src_lbs.to(device=device, dtype=torch.long)

			y = (y_cls_lbs, y_src_lbs)

			
			# TODO: early breaker
			if kconfig.tr.early_break and i == 3:
				print('[Break] by force for validation check')
				break




			# 
			if isinstance(model, dict):
				features_repr = model['feature_repr_model'](X)

				cls_output = model['cls_model'](features_repr)
				src_output = model['src_model'](features_repr)

			else:	
				output = model(X) #


			# pdb.set_trace()

			# Reset gradient
			reset_grad(optimizer)
 

			# Domain-Adversarial Training of Neural Networks: https://arxiv.org/abs/1505.07818
			if isinstance(loss_function, dict):
				cls_loss = loss_function['cls_loss'](cls_output, y_cls_lbs)
				src_loss = loss_function['src_loss'](src_output, y_src_lbs)

				feature_loss = cls_loss - src_loss

				# pdb.set_trace()


				feature_loss.backward()


				# print(f"\n[Wt] = {model['feature_repr_model'].fc1[0].weight[:10, ...]}")
				# print(f"\n[grad] = {model['feature_repr_model'].fc1[0].weight.grad[:10, ...]}")

				

				optimizer['feature_repr_opt'].step()

				print('[After] step')
				# print(f"\n[Wt] = {model['feature_repr_model'].fc1[0].weight[:10, ...]}")
				# print(f"\n[grad] = {model['feature_repr_model'].fc1[0].weight.grad[:10, ...]}")



				# print('[Check what happens to gradient]')


				reset_grad(optimizer)


				# pdb.set_trace()

				features_repr = features_repr.detach()

				cls_output = model['cls_model'](features_repr)
				src_output = model['src_model'](features_repr)


				# cls_output = model['cls_model'](features_repr)
				cls_loss = loss_function['cls_loss'](cls_output, y_cls_lbs)
				cls_loss.backward()

				optimizer['cls_opt'].step()



				# src_output = model['src_model'](features_repr)
				src_loss = loss_function['src_loss'](src_output, y_src_lbs)
				src_loss.backward()

				optimizer['src_opt'].step()


				loss = feature_loss + cls_loss + src_loss


			else:
				loss = loss_function(output, y)
				loss.backward()
				optimizer.step()




			# gap = 1
			# if counter > 1 and counter % gap == 0:
			# 	print()
			# 	print('\n\n\n[BBBBBBefore loss.backward()]')
			# 	print_wt_n_output(model, y, output, optimizer=optimizer)
				
			# loss.backward()
			
			# if counter > 1 and counter % gap == 0:
			# 	print('\n\n\n[AAAAAAfter loss.backward()]')
			# 	print_wt_n_output(model, y, output, optimizer=optimizer)
			
			# optimizer.step()
			
			# if counter > 1 and counter % gap == 0:
			# 	print('\n\n\n[AAAAAfter optimizer.step()]')
			# 	print_wt_n_output(model, y, output, optimizer=optimizer)
			# counter += 1

			# pdb.set_trace()
			# if isinstance(loss_function, dict):

			# 	# check <model['feature_repr_model']> grad before and after also 
			# 	# check <model['cls_model']>	
			# 	feature_loss.backward()


			# 	cls_loss.backward()
			# 	src_loss.backward()

			# else:
			# 	loss.backward()


			# set_momentum(0.90)


			train_cur_epoch_batchwise_loss.append(loss.item())
			train_all_epoch_batchwise_loss.append(loss.item())

			batch_run_msg = '\nEpoch: [%s/%s], Step: [%s/%s], InitialLR: %s, CurrentLR: %s, Loss: %s' \
							% (epoch + 1, total_epochs, i + 1, len(train_dataloader), start_lr, get_lr(), loss.item())
			print(batch_run_msg)
		#------------------ End of an Epoch ------------------ 
		
		# store average loss
		epoch_avg_loss = np.round(sum(train_cur_epoch_batchwise_loss) / (i + 1.0), 6)
		train_cur_epoch_batchwise_loss = []
		train_epoch_avg_loss_container.append(epoch_avg_loss)
		

		# 'Validation': xompute metrics the dataset for saving the models at checkpoints.
		if not (kconfig.tr.lr_search_flag or sanity_check):
			val_loss, val_report, f1_checker, auc_val = cal_loss_and_metric(model, dataloader['val'], loss_function, epoch+1)
			val_report['roc'] = 'Removed'
		

		# 'Validation': save model if certain condition is met on the computed metrics. 
		test_test_data = False
		accuracy = None 
		if not (kconfig.tr.lr_search_flag or sanity_check):
			val_report_container.append(val_report)  # ['epoch_' + str(epoch)] = val_report

			# Check point for which models will be saved
			val_avg_loss_container.append(val_loss)
			val_f1_container.append(f1_checker)	
			val_auc_container.append(auc_val)

			accuracy = val_report.get('accuracy', None)
			val_accuracy_container.append(accuracy)

			if np.round(val_loss, 4) <= np.round(min(val_avg_loss_container), 4):
				model = save_model(model, extra_extension='_minval') # + '_epoch_' + str(epoch))

			if np.round(auc_val, 4) >= np.round(max(val_auc_container), 4):
				model = save_model(model, extra_extension='_maxauc') # + '_epoch_' + str(epoch))
				test_test_data = True

			if np.round(f1_checker, 4) >= np.round(max(val_f1_container), 4):
				model = save_model(model, extra_extension='_maxf1') # + '_epoch_' + str(epoch))
				test_test_data = True



		# Save
		if epoch_avg_loss <= min(train_epoch_avg_loss_container):
			model = save_model(model, extra_extension='_mintrain')


		
		# Logger msg
		msg = '\n\n\n\n\nEpoch: [%s/%s], InitialLR: %s, CurrentLR= %s \n' \
			  '\n\n[Train] Average Epoch-wise Loss = %s \n' \
			  '\n\n********************************************************** [Validation]' \
			  '\n\n[Validation] Average Epoch-wise loss = %s \n' \
			  '\n\n[Validation] Report () = %s \n'\
			  '\n\n[Validation] F-Report = %s\n'\
			  '\n\n[Validation] Accuracy = %s\n'\
			  %(epoch+1, total_epochs, start_lr, get_lr(), train_epoch_avg_loss_container, val_avg_loss_container, None if not val_report_container else util.pretty(val_report_container[-1]), val_f1_container, val_accuracy_container)
		logger.info(msg); print(msg)


		# 'Test': compute metrics on the test dataset. Again, of course, only if it is present. 
		if not (kconfig.tr.lr_search_flag or sanity_check) and test_test_data and dataloader.get('test', False):
			test_loss, test_report, test_f1_checker, test_auc = cal_loss_and_metric(model, dataloader['test'], loss_function, epoch+1, model_type='test_set')
			
			test_report['roc'] = 'Removed'
			accuracy = test_report.get('accuracy', None)

			
			test_auc_container[epoch+1] = "{0:.3f}".format(round(test_auc, 4)) 
			test_f1_container[epoch+1] = "{0:.3f}".format(round(test_f1_checker, 4))
			test_accuracy_container[epoch+1] = "{}".format(accuracy)

			
			msg = '\n\n\n\n**********************************************************[Test]\n '\
				  '[Test] Report = {}' \
				  '\n\n[Test] fscore = {}' \
				  '\n\n[Test] AUC dict = {}' \
				  '\n\n[Test] F1-dict = {}'\
				  '\n\n[Test] Accuracy = {}'.format(util.pretty(test_report), test_f1_checker, test_auc_container, test_f1_container, test_accuracy_container)

			logger.info(msg); print(msg)

		
		# Strop training if the 'model' is already converged. 
		if epoch_avg_loss < 1e-6 or get_lr()[0] < 1e-11 or get_lr()[0] >= 10:
			msg = '\n\nAvg. Loss = {} or Current LR = {} thus stopping training'.format(epoch_avg_loss, get_lr())
			logger.info(msg)
			print(msg)
			break
			
		
		# Cyclic alteration of 'LR' during up and down steps movements.
		if kconfig.tr.lr_search_flag:
			# lr_scheduler.step(epoch + 1) # TODO: Only for estimating good learning rate
			if isinstance(lr_scheduler, dict):
				for key, lr_scheduler_eg in lr_scheduler.items():
					lr_scheduler_eg.step(epoch + 1)
		else:
			one_cycle_lr_setter(epoch + 1)

		# Time keeping for training epoch time. 
		end_time = time.time()
		end_datetime = datetime.now()
		msg = '\n\n[Time] taken for epoch({}) time = {}, datetime = {} \n\n'.format(epoch+1, end_time - start_time, end_datetime - start_datetime)
		logger.info(msg); print(msg)

	# ----------------- End of training process -----------------

	msg = '\n\n[Epoch Loss] = {}'.format(train_epoch_avg_loss_container)
	logger.info(msg); print(msg)

	
	# [LR]TODO: change for lr finder
	if kconfig.tr.lr_search_flag:
		losses = train_epoch_avg_loss_container
		plot_file_name = 'training_epoch_loss_for_lr_finder.png'
		title = 'Training Epoch Loss'
	else:
		losses = {'train': train_epoch_avg_loss_container, 'val': val_avg_loss_container}
		plot_file_name = 'training_vs_val_epoch_avg_loss.png'
		title= 'Training vs Validation Epoch Loss'
	plot_loss(losses=losses,
			plot_file_name=plot_file_name,
			title=title)
	plot_loss(losses=train_all_epoch_batchwise_loss, plot_file_name='training_batchwise.png', title='Training Batchwise Loss',
			xlabel='#Batchwise')
		

	# Save the model		
	model = save_model(model)


#----------------------------------------------------------------------------

def cal_loss_and_metric(model: torch.nn.Module, 
						dataloader: torch.utils.data.DataLoader, 
						loss_func: torch.nn.Module, 
						epoch_iter=0, 
						model_type='val_set'):
	"""Computes loss on val/test data and return a prepared metric report on that"""
	

	if isinstance(model, dict):
		for k, v in model.items():
			model[k] = v.eval()
	else:
		model = model.eval()

	# pdb.set_trace()
	
	total_loss = 0
	y_pred_all = None
	y_true_all = None
	with torch.no_grad():
		for i, (X, y) in enumerate(dataloader): # last one is true image size
			print('[Val/Test] batch i = ', i)
			loss = 0

			y_cls_lbs, y_src_lbs = y
			# print(f'[Class Labels Counts] = {y_cls_lbs.unique(return_counts=True)}', end='')
			# print(f'[Source Labels Counts] = {y_src_lbs.unique(return_counts=True) }')

			X = X.to(device=device, dtype=torch.float32) # or float is alias for float32


			y_cls_lbs = y_cls_lbs.to(device=device, dtype=torch.long)
			y_src_lbs = y_src_lbs.to(device=device, dtype=torch.long)


			# pdb.set_trace()


			if isinstance(model, dict):
				features_repr = model['feature_repr_model'](X)

				y_pred_cls = cls_output = model['cls_model'](features_repr)
				y_pred_src = src_output = model['src_model'](features_repr)

			else:	
				output = model(X) # 


			if isinstance(loss_func, dict):
				cls_loss = loss_func['cls_loss'](cls_output, y_cls_lbs)
				src_loss = loss_func['src_loss'](src_output, y_src_lbs)

				feature_loss = cls_loss - src_loss

				loss = feature_loss + cls_loss + src_loss

			else:
				loss = loss_func(output, y)


			y = (y_cls_lbs, y_src_lbs)



			total_loss += loss.item()


			# Because we need to only generate 'metric' on class labels i.e. object classes. It is not for the origin of data. 
			y_pred_all = y_pred_cls if y_pred_all is None else torch.cat((y_pred_all, y_pred_cls), dim=0)
			y_true_all = y_cls_lbs if y_true_all is None else torch.cat((y_true_all, y_cls_lbs), dim=0)



	# pdb.set_trace()
	
	report, f1_checker, auc_val = gen_metric_report(y_true_all, y_pred_all)

	if isinstance(model, dict):
		for k, v in model.items():
			model[k] = v.train()
	else:
		model = model.train()


	avg_loss =  np.round(total_loss / (i + 1.0), 6)
	
	if kconfig.img.num_cls_lbs > 2:
		plot_roc_curves_multiclass(report, epoch_iter, model_type=model_type)
	else:
		plot_roc_curves_binclass(report, epoch_iter, model_type=model_type)
	
	return avg_loss, report, f1_checker, auc_val  # avg_loss, score, model



#----------------------------------------------------------------------------

# Plot training loss
def plot_loss(losses, plot_file_name='training_loss.png', title='Training Loss', xlabel='Epochs'):
	fig = plt.figure()
	label_key = {'train': 'Training Loss', 'val': 'Validation Loss'}
	if isinstance(losses, dict):
		for k, v in losses.items():
			plt.plot(range(1, len(v)), v[1:], '-*', markersize=3, lw=1, alpha=0.6, label=label_key[k])	
	else:
		plt.plot(range(1, len(losses)+1), losses, '-*', markersize=3, lw=1, alpha=0.6)
	
	
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('BCE Loss')
	plt.legend(loc='upper right')
	full_path = os.path.join(util.get_results_dir(), plot_file_name)
	fig.tight_layout()  # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
	fig.savefig(full_path)
	plt.close(fig)  # clo




#----------------------------------------------------------------------------

def save_model(model, extra_extension=""):
	msg = '[Save] model extra_extension = {}'.format(extra_extension)
	logger.info(msg); print(msg)

	model_path = os.path.join(util.get_models_dir(), util.get_custom_model_name(extra_extension))

	if isinstance(model, dict):
		model_dict = {}
		for mdl_name, mdl in model.items():
			if next(mdl.parameters()).is_cuda:
				mdl =  mdl.cpu().float()
			model_dict[mdl_name] = mdl.state_dict()
	else:
		if next(model.parameters()).is_cuda:
			model = model.cpu().float()

		model_dict = model.state_dict()


	torch.save(model_dict, model_path)
	
	ret_mdl = {}
	if isinstance(model, dict):
		for k, mdl in model.items():
			ret_mdl[k] = mdl.to(device)
	else:
		ret_mdl = model.to(device)


	return ret_mdl


#----------------------------------------------------------------------------

# Pre-requisite setup for training process
def init_for_training(train_data_info, val_data_info, test_data_info, sanity_check=False, use_pretrained=False):
	"""Setup all the pre-requisites for complete training of the model 

		Parameters:
		-----------
			train_data_info (dict): Arguments needed to setup train dataset such datapath etc.
			val_data_info (dict): Arguments needed to setup val dataset such datapath etc.
			test_data_info (dict): Arguments needed to check performance on test set
			sanity_check (bool): pass the boolean to the method <train_network> to indicate if it is sanity check or full training

		Returns:
		--------
			None: Only works as setup for the training of the model
	"""
	msg = '\n\n[Train] data info = {}\n\n[Validation] data info = {}\n\n[SanityCheck] = {}'.format(train_data_info, val_data_info, sanity_check)
	logger.info(msg), print(msg)
	
	train_params = {}
	# [LR]
	if kconfig.tr.lr_search_flag:
		start_lr, end_lr, epochs = *kconfig.lr.lr_search, kconfig.epochs.lr_search
	else:
		start_lr, end_lr, epochs =  *kconfig.lr.train, kconfig.epochs.train # 1e-3, 9e-3, 90 # 90 5e-5, 2e-4, 70  
	train_params['start_lr'] = start_lr = start_lr
	train_params['end_lr'] = end_lr
	train_params['num_epochs'] = epochs


	use_batchnorm = True # TODO: batchnorm
	dropout = 0.0
	if sanity_check or kconfig.tr.lr_search_flag:
		weight_decay = kconfig.hp.lr_search.weight_decay
		dropout = kconfig.hp.lr_search.dropout
	else:
		weight_decay = kconfig.hp.train.weight_decay # 1e-4 # 1e-6
		dropout = kconfig.hp.train.dropout # 0.5 # might not be needed
	


	dataset = {}
	dataset['train'] = ImageDataset(**train_data_info)
	dataset['val'] = ImageDataset(**val_data_info)
	if test_data_info is not None:
		dataset['test'] = ImageDataset(**test_data_info)



	dataloader = {}
	# pdb.set_trace()
	dataloader['train'] = DataLoader(dataset=dataset['train'], batch_size=kconfig.tr.batch_size, shuffle=False)
	dataloader['val'] = DataLoader(dataset=dataset['val'], batch_size=kconfig.val.batch_size)
	if test_data_info is not None:
		dataloader['test'] = DataLoader(dataset=dataset['test'], batch_size=kconfig.test.batch_size)

	train_params['dataloader'] = dataloader

	# NN Args
	net_args = {}
	net_args['in_channels'] = 3
	
	# net_args['num_cls_lbs'] = kconfig.img.num_cls_lbs
	# net_args['num_src_lbs'] = kconfig.img.num_src_lbs
	
	net_args['model_img_size'] = kconfig.img.size
	net_args['nonlinearity_function'] = None # nn.LeakyReLU()
	net_args['use_batchnorm'] = kconfig.tr.use_batchnorm # False
	net_args['dropout'] = dropout
	# net_args['use_batchnorm'] = use_batchnorm

	model = {}
	if kconfig.tr.use_pretrained_flag:
		# net_args['eps'] = 1e-3
		model = PretrainedADAConvNet(**net_args)
	else:
		model['feature_repr_model'] = ADAConvNet(**net_args)

		feature_repr_len = ADAConvNet.compute_feature_repr_len( kconfig.img.size)

		msg = f'[feature_repr_len] = {feature_repr_len}'

		print(msg); logger.info(msg);

		model['cls_model'] = ADAConvNetClsLabel(feature_repr_len, kconfig.img.num_cls_lbs)
		model['src_model'] = ADAConvNetSrcLabel(feature_repr_len, kconfig.img.num_src_lbs)


	# pdb.set_trace()
	if isinstance(model, dict):
		for k, v in model.items():
			model[k] = v.to(device)
	else:
		 model = model.to(device)

	train_params['model'] = model


	# Loss Args
	loss_args = {}
	loss_args['cls_lbs_weight'] = kconfig.img.cls_lbs_weight
	loss_args['src_lbs_weight'] = kconfig.img.src_lbs_weight

	# pdb.set_trace()
	loss_args['reduction'] = kconfig.loss.reduction
	

	loss_function = {'cls_loss': ADAClsLabelLoss(kconfig.img.cls_lbs_weight, kconfig.loss.reduction), 
					 'src_loss': ADASrcLabelLoss(kconfig.img.src_lbs_weight, kconfig.loss.reduction)
					 }

	if isinstance(loss_function, dict):
		for k, v in loss_function.items():
			loss_function[k] = v.to(device)
	else:
		loss_function = loss_function.to(device)


	train_params['loss_function'] = loss_function


	optimizer = {}

	momentum = 0.95

	if isinstance(model, dict):
		for k, v in model.items():
			opt_name = '_'.join(k.split('_')[:-1]) + '_opt'
			print(f'[Key] = {k}, Opt Name = {opt_name}, Net Name = {v.__class__.__name__}')
			optimizer[opt_name] = Optimizer.adam_optimizer(params=v.parameters(), lr=start_lr, weight_decay=weight_decay)
			# rmsprop_optimizer(params=v.parameters(), lr=start_lr, weight_decay=weight_decay)
			# sgd_optimizer(params=v.parameters(), lr=start_lr, weight_decay=weight_decay, momentum=momentum)
	else:
		optimizer = Optimizer.sgd_optimizer(params=model.parameters(), lr=start_lr, weight_decay=weight_decay, momentum=momentum)

	# Optimizer.sgd_optimizer(params=v.parameters(), lr=start_lr, weight_decay=weight_decay, momentum=momentum)
	# optimizer = Optimizer.adam_optimizer(params=model.parameters(), lr=start_lr, weight_decay=weight_decay)
	# optimizer = Optimizer.rmsprop_optimizer(params=model.parameters(), lr=start_lr, weight_decay=weight_decay)


	train_params['optimizer'] = optimizer
	train_params['sanity_check'] = sanity_check
	

	# Train the network
	train_network(**train_params)



#----------------------------------------------------------------------------

def main(sanity_check=False):

	np.random.seed(99)
	torch.manual_seed(99)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(99)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	print('Pid = ', os.getpid())



	util.set_trained_model_name(ext_cmt='_paper') # ext_cmt = ['scratch', 'pretrained_resnet50']


	# TODO: use all transformer for full training
	train_data_args = {}
	train_data_args['transformers'] = [] # [RandomNoiseImage(), MirrorImage(), InvertVerticallyImage(), MirrorAndInvertVerticallyImage(), Rotate90Image(), Rotate270Image(), ChangeSaturationImage(), ChangeLuminescenceImage()] # ChangeHSLImage(), ChangeHueImage()

	# root_data_dir, img_info_datapath, transformers=None, model_img_size=(64, 64), normalizer_dict_path=None, has_label=True):
	

	# Dataset Args
	data_dir =  kconfig.img.data_dir # ''

	data_info_args = {}
	data_info_args['root_data_dir'] = util.get_data_dir(data_dir)
	data_info_args['model_img_size'] = kconfig.img.size
	data_info_args['normalizer_dict_path'] = util.get_normalization_info_pickle_path()
	data_info_args['has_label'] = True



	train_data_info = {'img_info_datapath': util.get_train_info_datapath(), **data_info_args, **train_data_args}
	val_data_info = {'img_info_datapath': util.get_val_info_datapath(), **data_info_args}
	test_data_info = None # {'img_info_datapath': util.get_test_info_datapath(), **data_info_args}
	



	msg = '[Datapath] \nTrain = {}, \nValidation = {}'.format(train_data_info, val_data_info)
	logger.info(msg); print(msg)
	init_for_training(train_data_info=train_data_info, val_data_info=val_data_info, test_data_info=test_data_info, sanity_check=sanity_check)
	


#----------------------------------------------------------------------------

def get_arguments_parser(eval_steps):
	"""Argument parser for predition"""
	description = 	'Provide arguments for training'
					
	parser = argparse.ArgumentParser(description=description)


	parser.add_argument('-e', '--eval-steps', type=int, default=eval_steps, 
		help='Provide how many evaluation steps has to be performed during training.', required=False)


	return parser



#----------------------------------------------------------------------------

if __name__ == '__main__':
	print('Trainer')

	
	# arg_parser = get_arguments_parser(5)
	# arg_parser = arg_parser.parse_args()
	# eval_steps = arg_parser.eval_steps 


	# msg = '[Args]: \neval_steps = {}'.format(eval_steps)
	# logger.info(msg);print(msg)

	# pdb.set_trace()
	# Sanity check of the model for learning
	sanity_check = kconfig.tr.sanity_check_flag
	main(sanity_check=sanity_check)
	logger.info('\n\n********************** Training Complete **********************\n\n')




