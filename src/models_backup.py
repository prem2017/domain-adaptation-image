# -*- coding: utf-8 -*-


import os
import pdb

import torch
import torch.nn as nn
import torchvision.models as models

from . import util 

# https://pytorch.org/docs/stable/torchvision/models.html
# models.resnet50((pretrained=True, progress=True)), resnet18, inception_v3 


#----------------------------------------------------------------------------

device = util.get_training_device()


#----------------------------------------------------------------------------

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


#----------------------------------------------------------------------------

class View(nn.Module):
	"""docstring for View"""
	def __init__(self, *args, **kwargs):
		super(View, self).__init__()

	def forward(self, X):
		X = X.view(X.shape[0], -1)
		return X



#----------------------------------------------------------------------------Works

# Model
class ADAConvNet(nn.Module):
	"""ADAConvNet for classification"""
	def __init__(self, num_cls_lbs=11, num_src_lbs=3, in_channels=3, model_img_size=(64, 64), nonlinearity_function=None, use_batchnorm=False, dropout=0.5):
		"""
		  TODO: change architecture for class and source labels so that they adversarially learn from each other. 
		"""


		super(ADAConvNet, self).__init__()
		self.width, self.height = model_img_size

		if nonlinearity_function is None:
			self.nonlinearity_function = nn.ReLU()
		else:
			self.nonlinearity_function = nonlinearity_function

		self.use_batchnorm = use_batchnorm
		self.dropout = dropout

		self.accumulate_downsizing = 1
		
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
			self.nonlinearity_function,
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
			self.nonlinearity_function)
		
		self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2); self.accumulate_downsizing *= (2*2) # Kernel_size*stride
		
		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
			self.nonlinearity_function)
		
		self.layer3 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=5, stride=1, padding=2)
		# output size WxHx2
		
		self.fc1_units = fc1_units = int((self.width * self.height * 2) / self.accumulate_downsizing)
		
		self.fc1 = nn.Sequential(
			nn.Linear(fc1_units, fc1_units // 2),
			self.nonlinearity_function,
			nn.Dropout(p=dropout),
			nn.Linear(fc1_units //2, fc1_units // 4),
			self.nonlinearity_function,
			nn.Dropout(p=dropout),
			)
		
		self.fc_cls_lbs = nn.Linear(fc1_units // 4, num_cls_lbs)
		self.fc_src_lbs = nn.Linear(fc1_units // 4, num_src_lbs)

	
	
	
	def forward(self, x):
		out = self.layer1(x)
		out = self.down_sample(out)
		
		out = self.layer2(out)
		out = self.layer3(out)
	
		out = out.view(-1, self.num_flat_features(out))
		features_repr = self.fc1(out)

		out_cls_lbs = self.fc_cls_lbs(features_repr)
		out_src_lbs = self.fc_src_lbs(features_repr)

		return out_cls_lbs, out_src_lbs

		
	
	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features




#----------------------------------------------------------------------------

# Model
class ADAConvNet(nn.Module):
	"""ADAConvNet for classification"""
	def __init__(self, num_cls_lbs=11, num_src_lbs=3, in_channels=3, model_img_size=(64, 64), nonlinearity_function=None, use_batchnorm=False, dropout=0.5):
		"""
		  TODO: change architecture for class and source labels so that they adversarially learn from each other. 
		"""


		super(ADAConvNet, self).__init__()
		self.width, self.height = model_img_size

		if nonlinearity_function is None:
			self.nonlinearity_function = nn.ReLU()
		else:
			self.nonlinearity_function = nonlinearity_function

		self.use_batchnorm = use_batchnorm
		self.dropout = dropout

		self.accumulate_downsizing = 1
		
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
			self.nonlinearity_function,
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
			self.nonlinearity_function)
		
		self.down_sample1 = nn.MaxPool2d(kernel_size=2, stride=2); self.accumulate_downsizing *= (2*2) # Kernel_size*stride
		
		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
			self.nonlinearity_function,
			nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
			self.nonlinearity_function)

		self.down_sample2 = nn.MaxPool2d(kernel_size=2, stride=2); self.accumulate_downsizing *= (2*2) # Kernel_size*stride

		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
			self.nonlinearity_function)
		
		self.layer4 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, stride=1, padding=2)
		# output size WxHx2
		# output size WxHx2
		
		self.fc1_units = fc1_units = int((self.width * self.height * 2) / self.accumulate_downsizing)
		
		self.fc1 = nn.Sequential(
			nn.Linear(fc1_units, fc1_units // 2),
			self.nonlinearity_function,
			nn.Dropout(p=dropout),
			nn.Linear(fc1_units //2, fc1_units // 4),
			self.nonlinearity_function,
			nn.Dropout(p=dropout),
			)
		
		self.fc_cls_lbs = nn.Linear(fc1_units // 4, num_cls_lbs)
		self.fc_src_lbs = nn.Linear(fc1_units // 4, num_src_lbs)

	
	
	
	def forward(self, x):
		out = self.layer1(x)
		out = self.down_sample1(out)

		# pdb.set_trace()

		out = self.layer2(out)
		out = self.down_sample2(out)

		out = self.layer3(out)
		out = self.layer4(out)

		out = out.view(-1, self.num_flat_features(out))




		features_repr = self.fc1(out)
		out_cls_lbs = self.fc_cls_lbs(features_repr)
		out_src_lbs = self.fc_src_lbs(features_repr)


		return out_cls_lbs, out_src_lbs

		
	
	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

