# -*- coding: utf-8 -*-


# ©Prem Prakash
# Report generator helper module


import os
import pdb

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc # (y_true, y_score)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn


from . import util 
from .util import kconfig
from .util import logger





#----------------------------------------------------------------------------

# TODO: 
def compute_label_and_prob(pred: torch.Tensor, th=0.5):
	"""Computes probability  and label/class"""
	# pdb.set_trace()
	bin_clf = True
	if pred.ndim > 1:
		bin_clf = False
		softmax = nn.Softmax(dim=1)
		pred_prob = softmax(pred).data
		# pred_prob = pred_prob[:, 1]
		pred_prob_max, pred_label_max = pred_prob.max(dim=1, keepdim=True) 
		if pred.shape[1] == 2: # It means there are only two classes so computer for only class-1 i.e. positive
			bin_clf = True
			pred = pred[:, 1]

	if bin_clf:
		pred_prob_max = pred.sigmoid().reshape(-1, 1)
		label = deepcopy(pred_prob_max)
		label[label >= th] = 1
		label[label <= th] = 0
		pred_label_max = torch.as_tensor(label, dtype=int)

	return pred_prob_max, pred_label_max


#----------------------------------------------------------------------------

def compute_prob_all(pred: torch.Tensor):
	"""Computes probability from predicted value"""
	# pdb.set_trace()
	softmax = nn.Softmax(dim=1)
	pred_prob_all = softmax(pred).data
	
	return pred_prob_all


#----------------------------------------------------------------------------

def gen_metric_report(ytrue: torch.Tensor, ypred: torch.Tensor):
	"""Generated report from the predcition such as classification-report, confusion-matrix, F1-score, ROC """
	# pdb.set_trace()
	ytrue = ytrue.contiguous()
	ypred = ypred.contiguous()

	ypred_prob, ypred_label = compute_label_and_prob(ypred)
	ypred_label = ypred_label.cpu().numpy()
	ypred_prob = ypred_prob.cpu().numpy()

	ytrue = ytrue.cpu().numpy()
	class_names = list(kconfig.img.labels_dict.values())  # 'cl_name1', 'cl_name2' and so on
	label_names =  list(kconfig.img.labels_dict.keys()) # 0, 1, 2 so on
	num_classes = len(class_names)



	report = {}	
	report['accuracy'] = accuracy_score(y_true=ytrue, y_pred=ypred_label)
	report['clf_report'] = classification_report(y_true=ytrue, y_pred=ypred_label, target_names=class_names)
	
	if num_classes == 2:
		f1_checker = report['f1_score'] = f1_score(y_true=ytrue, y_pred=ypred_label) # only for binary class
	else:
		f1_checker = report['f1_score_micro'] = f1_score(y_true=ytrue, y_pred=ypred_label, average='micro')
		report['f1_score_macro'] = f1_score(y_true=ytrue, y_pred=ypred_label, average='macro')

	# pdb.set_trace()
	report['conf_mat'] = confusion_matrix(y_true=ytrue, y_pred=ypred_label, labels=label_names)

	auc_val  = 0
	if num_classes == 2: 
		fpr, tpr, ths = roc_curve(y_true=ytrue, y_score=ypred_prob)
		report['roc'] = fpr, tpr, ths
		auc_val = report['auc'] = auc(fpr, tpr)
	else:
		y_true = label_binarize(ytrue, classes=label_names)
		
		pred_prob_all = compute_prob_all(ypred)
		pred_prob_all = pred_prob_all.cpu().numpy()

		roc_dict = {}
		auc_dict = {}

		for i, label_name in enumerate(label_names):
			fpr, tpr, ths = roc_curve(y_true=y_true[:, i], y_score=pred_prob_all[:, i])
			auc_val = auc(fpr, tpr)
			roc_dict[label_name] = fpr, tpr, ths, auc_val
			auc_dict[label_name] = auc_val


		fpr, tpr, ths  = roc_curve(y_true=y_true.ravel(), y_score=pred_prob_all.ravel()) 
		auc_val = report['auc'] = auc(fpr, tpr)
		roc_dict['micro-average'] = fpr, tpr, ths, auc_val
		auc_dict['micro'] = auc_val

		report['roc'] = roc_dict
		report['auc_dict'] = auc_dict
		

	return report, f1_checker, auc_val


#----------------------------------------------------------------------------

def plot_roc_curves_binclass(roc_data, epoch_iter, model_type='val_set', ext_cmt=''):
	"""Plots ROC curve after each iteration """
	fpr, tpr, ths = roc_data['roc']

	tpr_fpr = tpr - fpr
	optimal_idx = np.argmax(tpr_fpr)
	max_tpr_fpr = tpr_fpr[optimal_idx]

	optimal_fpr, optimal_tpr, optimal_threshold = fpr[optimal_idx], tpr[optimal_idx], ths[optimal_idx]

	auc_val = roc_data['auc']

	msg = f'\n\n[{model_type}] on ROC best (tpr-fpr) = {max_tpr_fpr}, optimal_fpr = {optimal_fpr},optimal_tpr = {optimal_tpr}, optimal-threshold = {optimal_threshold}, AUC-ROC = {auc_val}\n\n'
	logger.info(msg); print(msg)

	fig = plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color=color, lw=lw, label=f'ROC curve for {class_name} (area = {round(auc_val, 2)})\nMax(tpr-fpr) = {max_tpr_fpr}\nth={round(optimal_threshold, 9)})')
	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.plot(optimal_fpr, optimal_tpr, color='green', marker='*', markersize=9)

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])

	plt.xlabel('False Positive Rate (1-Specificity)')
	plt.ylabel('True Positive Rate (Recall)')
	plt.title('ROC Curve')
	plt.legend(loc='lower right')

	base_path = util.get_results_dir(model_type)
	if not os.path.exists(base_path):
		os.mkdir(base_path)
	full_path = os.path.join(base_path, str(epoch_iter) + '_' + ext_cmt + '_bin' + '.png')
	fig.tight_layout()
	plt.savefig(full_path)
	plt.close(fig)


#----------------------------------------------------------------------------

def plot_roc_curves_multiclass(roc_data, epoch_iter, model_type='val_set', ext_cmt=''):
	"""Plots ROC curve after each iteration """
	roc_dict = roc_data['roc']

	class_names = kconfig.img.labels_dict.values()  
	num_classes = len(class_names)
	
	fig = plt.figure()
	lw = 2
	color_dict = kconfig.img.colors_dict_for_plotting 
	class_label_dict =  kconfig.img.labels_dict

	# pdb.set_trace()
	for label_key, roc_vals in roc_dict.items():
		if color_dict.get(label_key, None) is None:
			continue # Ignore for which color is not available

		color = color_dict[label_key]
		class_name = class_label_dict.get(label_key, label_key) 


		fpr, tpr, ths, auc_val = roc_vals

		tpr_fpr = tpr - fpr
		optimal_idx = np.argmax(tpr_fpr)
		max_tpr_fpr = tpr_fpr[optimal_idx]
		optimal_fpr, optimal_tpr, optimal_threshold = fpr[optimal_idx], tpr[optimal_idx], ths[optimal_idx]

		msg = f'\n\n[{model_type}] on ROC best (tpr-fpr) = {max_tpr_fpr}, optimal_fpr = {optimal_fpr},optimal_tpr = {optimal_tpr}, optimal-threshold = {optimal_threshold}, AUC-ROC = {auc_val}\n\n'
		logger.info(msg); print(msg)

		max_tpr_fpr = '{0:.3f}'.format(max_tpr_fpr)
		optimal_threshold = '{0:.3f}'.format(optimal_threshold)
		plt.plot(fpr, tpr, color=color, lw=lw, label=f'ROC curve for {class_name} (area = {round(auc_val, 2)})\nMax(tpr-fpr) = {max_tpr_fpr}, th={optimal_threshold})')
		plt.plot(optimal_fpr, optimal_tpr, color='green', marker='*', markersize=9)


	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])

	plt.xlabel('False Positive Rate (1-Specificity)')
	plt.ylabel('True Positive Rate (Recall)')
	plt.title('ROC Curve for all classes')
	plt.legend(loc='lower right')

	base_path = util.get_results_dir(model_type)
	if not os.path.exists(base_path):
		os.mkdir(base_path)
	full_path = os.path.join(base_path, str(epoch_iter) + '_' + ext_cmt + '_multi' + '.png')

	fig.tight_layout()
	plt.savefig(full_path)
	plt.close(fig)