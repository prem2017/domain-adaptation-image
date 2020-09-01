# -*- coding: utf-8 -*-


import os
import math
import pdb

import torch
import torch.nn as nn


from .util import logger


#----------------------------------------------------------------------------


def print_nan_loss(loss_val, output, y_target, ltyp='Class'):

	if math.isnan(loss_val.item()):
		print(f'\n\n[{ltyp}] =>')
		print('		Loss = ', loss_val.item()); logger.info(loss_val.item())
		print('		Output = ', output); logger.info(output)
		print('		y_target = ', y_target); logger.info(y_target)




#----------------------------------------------------------------------------

class ADALoss(nn.Module):
	"""docstring for ADALoss
			Loss from Class Labels and Source Labels 
			Source is also used as adversarial so half the time they are correctly labelled 
			And at the other times it is labelled to confuse the network thus adversarially training the network. 


	"""
	
	def __init__(self, cls_lbs_weight=None, src_lbs_weight=None,  reduction='mean'):
		"""
			cls_lbs_weight & src_lbs_weight
				weight (Tensor, optional): a manual rescaling weight given to each class. If given, has to be a Tensor of size `C`


		"""
 

		super(ADALoss, self).__init__()
		self.cls_lbs_weight = cls_lbs_weight
		self.src_lbs_weight = src_lbs_weight
		self.reduction = reduction

		self.cls_lb_loss_func = nn.CrossEntropyLoss(weight=self.cls_lbs_weight, reduction=self.reduction)
		self.src_lb_loss_func = nn.CrossEntropyLoss(weight=self.src_lbs_weight, reduction=self.reduction)




	
	def forward(self, output, y_target):
		output_cls_lbs, output_src_lbs = output
		y_target_cls_lbs, y_target_src_lbs = y_target

		# pdb.set_trace()
		loss_cls_lb = self.cls_lb_loss_func(output_cls_lbs, y_target_cls_lbs)
		loss_src_lb = self.src_lb_loss_func(output_src_lbs, y_target_src_lbs)
		
		print_nan_loss(loss_cls_lb, output_cls_lbs, y_target_cls_lbs, 'Class')
		print_nan_loss(loss_src_lb, output_src_lbs, y_target_src_lbs, 'Source')

		loss = loss_cls_lb + loss_src_lb
		
		return loss



#----------------------------------------------------------------------------

# Class Loss
class ADAClsLabelLoss(nn.Module):
	"""docstring for ADAClsLabelLoss"""

	def __init__(self, cls_lbs_weight=None,  reduction='mean'):
		super(ADAClsLabelLoss, self).__init__()
		
		self.cls_lbs_weight = cls_lbs_weight
		self.reduction = reduction

		self.cls_lb_loss_func = nn.CrossEntropyLoss(weight=self.cls_lbs_weight, reduction=self.reduction)



	def forward(self, output, y_target):

		loss = self.cls_lb_loss_func(output, y_target)

		print_nan_loss(loss, output, y_target, 'Class')

		return loss



		

#----------------------------------------------------------------------------

# Source Loss
class ADASrcLabelLoss(nn.Module):
	"""docstring for ADASrcLabelLoss"""

	def __init__(self, src_lbs_weight=None, reduction='mean'):
		super(ADASrcLabelLoss, self).__init__()
		
		self.src_lbs_weight = src_lbs_weight
		self.reduction = reduction

		self.src_lb_loss_func = nn.CrossEntropyLoss(weight=self.src_lbs_weight, reduction=self.reduction)



	def forward(self, output, y_target):

		loss = self.src_lb_loss_func(output, y_target)

		print_nan_loss(loss, output, y_target, 'Class')

		return loss








