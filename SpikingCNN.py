import collections
import sonnet as snt
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os 

from utils import lif_dynamic

#pass an image through the network without an error
class SpikingCNN(tf.keras.Model):
	"""
	SpikingCNN is a custom convolutional neural network model for spiking neural networks.
	It consists of two convolutional layers with optional additional layers for binary activation (ba) or average binary activation (avg_ba).

	Args:
		n_kernel_1 (int): Number of kernels in the first convolutional layer.
		n_filter_1 (int): Number of filters in the first convolutional layer.
		stride_1 (int): Stride size for the first convolutional layer
		tau (float): Decay constant for the spiking neurons.
		thr (float): Threshold value for the spiking neurons.

	Attributes:
		n_w_1 (int): Width of the output feature map after the first convolutional layer.
		n_w_2 (int): Width of the output feature map after the second convolutional layer.
		conv_layer_1_1 (tf.keras.layers.Conv2D): First convolutional layer with binary activation.
		conv_layer_2 (tf.keras.layers.Conv2D): Second convolutional layer.
	"""

	def __init__(self, n_kernel_1=8, n_filter_1=16, stride_1=4, n_kernel_2=8, n_filter_2=32, stride_2=4, ba=False,
				 avg_ba=False, ba_config=None, tau=1, thr=1., avg_pool_1_stride=4, avg_pool_1_k=8, avg_pool_2_stride=2,
				 avg_pool_2_k=4):

		super(SpikingCNN, self).__init__()
		self.decay = np.exp(-1 / tau)
		self.v_th = thr
		self.n_filters_1 = n_filter_1
		self.n_filters_2 = n_filter_2
		self.ba = ba
		self.avg_ba = avg_ba
		self.n_w_1 = (128 - n_kernel_1) // stride_1 + 2
		self.n_w_2 = (self.n_w_1 - n_kernel_2) // stride_2 + 1
		self.n_kernel_1 = n_kernel_1
		self.stride_1 = stride_1
		self.n_kernel_2 = n_kernel_2
		self.stride_2 = stride_2
		self.avg_pool_1_stride = avg_pool_1_stride
		self.avg_pool_2_stride = avg_pool_2_stride
		self.avg_pool_1_k = avg_pool_1_k
		self.avg_pool_2_k = avg_pool_2_k

		self.n_avg_1 = (self.n_w_1 - avg_pool_1_k) // avg_pool_1_stride + 1
		self.n_avg_2 = (self.n_w_2 - avg_pool_2_k) // avg_pool_2_stride + 1

		self.ba_filters_1_1 = 16
		self.ba_kernel_1_1 = 8
		self.ba_stride_1_1 = 4
		self.ba_filters_1_2 = 32
		self.ba_kernel_1_2 = 4
		self.ba_stride_1_2 = 2
		self.ba_filters_2 = 32
		self.ba_kernel_2 = 4
		self.ba_stride_2 = 2

		self.n_ba_1 = (self.n_w_1 - self.ba_kernel_1_1) // self.ba_stride_1_1 + 1
		self.n_ba_2 = (self.n_w_2 - self.ba_kernel_2) // self.ba_stride_2 + 1

		self.conv_layer_1_1 = tf.keras.layers.Conv2D(
			self.ba_filters_1_1,
			self.ba_kernel_1_1,
			strides=self.ba_stride_1_1,
			padding='valid',
			trainable=True)

		self.conv_layer_2 = tf.keras.layers.Conv2D(
			self.n_filters_2,
			self.n_kernel_2,
			strides=self.stride_2,
			padding='valid',
			trainable=True)

	@property
	def output_size(self):
		"""
		Returns the output sizes of the convolutional layers based on the configuration.

		Returns:
			tuple: Output sizes of the convolutional layers.
		"""
		if self.ba:
			return self.n_w_1 * self.n_w_1 * self.n_filters_1, \
				   self.n_w_1 * self.n_w_1 * self.n_filters_1, \
				   self.n_w_2 * self.n_w_2 * self.n_filters_2, \
				   self.n_w_2 * self.n_w_2 * self.n_filters_2, \
				   self.n_ba_1 * self.n_ba_1 * self.ba_filters_1_1, \
				   self.n_ba_2 * self.n_ba_2 * self.ba_filters_2
		if self.avg_ba:
			return self.n_w_1 * self.n_w_1 * self.n_filters_1, \
				   self.n_w_1 * self.n_w_1 * self.n_filters_1, \
				   self.n_w_2 * self.n_w_2 * self.n_filters_2, \
				   self.n_w_2 * self.n_w_2 * self.n_filters_2, \
				   self.n_avg_1 * self.n_avg_1 * self.n_filters_1, \
				   self.n_avg_2 * self.n_avg_2 * self.n_filters_2
		return self.n_w_1 * self.n_w_1 * self.n_filters_1, \
			   self.n_w_1 * self.n_w_1 * self.n_filters_1, \
			   self.n_w_2 * self.n_w_2 * self.n_filters_2, \
			   self.n_w_2 * self.n_w_2 * self.n_filters_2

	def zero_state(self, batch_size, dtype):
		"""
		Returns a zero state for the convolutional layers.

		Args:
			batch_size (int): Batch size.
			dtype: Data type.

		Returns:
			tuple: Zero state for the convolutional layers.
		"""
		return tf.zeros((batch_size, self.n_w_1, self.n_w_1, self.n_filters_1), dtype), \
			   tf.zeros((batch_size, self.n_w_2, self.n_w_2, self.n_filters_2), dtype)

	@property
	def state_size(self):
		"""
		Returns the state size of the convolutional layers.

		Returns:
			tuple: State size of the convolutional layers.
		"""
		return (self.n_w_1, self.n_w_1, self.n_filters_1), (self.n_w_2, self.n_w_2, self.n_filters_2)

	def __call__(self, inputs, state):
		"""
		Takes in a batch of images and returns the output of the convolutional layers.

		Args:
			inputs: Batch of images.
				Input tensor in the format: [batch size, width, height, channels].
			state: Tuple of the two convolutional layers.

		Returns:
			tuple: Output of the convolutional layers.
		"""
		# Convolutional layer 1
		v_conv_1, z_conv_1 = lif_dynamic(state[0], inputs, self.decay, self.v_th, 1.)

		# Convolutional layer 2
		i_conv_2 = self.conv_layer_2(z_conv_1)

		v_conv_2, z_conv_2 = lif_dynamic(state[1], i_conv_2, self.decay, self.v_th, 1.)

		c1_r = None
		c2_r = None

		new_state = (v_conv_1, v_conv_2)
		if self.ba or self.avg_ba:
			return (tf.reshape(z_conv_1, (-1, self.n_w_1 * self.n_w_1 * self.n_filters_1)),
					tf.reshape(v_conv_1, (-1, self.n_w_1 * self.n_w_1 * self.n_filters_1)),
					tf.reshape(z_conv_2, (-1, self.n_w_2 * self.n_w_2 * self.n_filters_2)),
					tf.reshape(v_conv_2, (-1, self.n_w_2 * self.n_w_2 * self.n_filters_2)), c1_r, c2_r), new_state
		return (tf.reshape(z_conv_1, (-1, self.n_w_1 * self.n_w_1 * self.n_filters_1)),
				tf.reshape(v_conv_1, (-1, self.n_w_1 * self.n_w_1 * self.n_filters_1)),
				tf.reshape(z_conv_2, (-1, self.n_w_2 * self.n_w_2 * self.n_filters_2)),
				tf.reshape(v_conv_2, (-1, self.n_w_2 * self.n_w_2 * self.n_filters_2))), new_state


