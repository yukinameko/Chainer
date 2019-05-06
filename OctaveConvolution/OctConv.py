import chainer
import chainer.links as L
import numpy as np

def isNone(n):
	return n == None

class OctConv2dBase(chainer.Chain):
	def __init__(self, in_channels, out_channels, in_alpha=0, out_alpha=0, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, *, dilate=1, groups=1):
		super(OctConv2dBase, self).__init__()
		with self.init_scope():
			in_low_channels = None
			in_high_channels = None

			if in_channels != None:
				in_low_channels = int(in_channels*in_alpha)
				in_high_channels = in_channels - in_low_channels

			out_low_channels = int(out_channels*out_alpha)
			out_high_channels = out_channels - out_low_channels

			if isNone(in_high_channels) or in_high_channels:
				if out_high_channels:
					self.conv_high_to_high = L.Convolution2D(in_high_channels, out_high_channels, ksize=ksize, stride=stride, pad=pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate, groups=groups)

				if out_low_channels:
					self.conv_high_to_low = L.Convolution2D(in_high_channels, out_low_channels, ksize=ksize, stride=stride, pad=pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate, groups=groups)

			if isNone(in_low_channels) or in_low_channels:
				if out_high_channels:
					self.conv_low_to_high = L.Convolution2D(in_low_channels, out_high_channels, ksize=ksize, stride=stride, pad=pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate, groups=groups)

				if out_low_channels:
					self.conv_low_to_low = L.Convolution2D(in_low_channels, out_low_channels, ksize=ksize, stride=stride, pad=pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate, groups=groups)


class OctConv2d(OctConv2dBase):
	"""docstring for OctConv2d"""
	def __call__(self, x):
		high, low = x

		high_to_high = self.conv_high_to_high(high)
		low_to_high = self.conv_low_to_high(F.avg_pooling_2d(low))

		high_to_low = self.conv_high_to_low(high)
		low_to_low = self.conv_low_to_low(low)

		out_high = high_to_high + low_to_high
		out_low = high_to_low + low_to_low

		return np.array([out_high, out_low])

class OctConv2dIn(OctConv2dBase):
	"""docstring for OctConv2dIn"""
	def __init__(self, in_channels, out_channels, out_alpha=0, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, *, dilate=1, groups=1):
		super(OctConv2dIn, self).__init__(in_channels, out_channels, in_alpha=0, out_alpha=out_alpha, ksize=ksize, stride=stride, pad=pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate, groups=groups)
	
	def __call__(self, x):
		high_to_high = self.conv_high_to_high(x)

		high_to_low = self.conv_high_to_low(x)

		out_high = high_to_high
		out_low = high_to_low

		return np.array([out_high, out_low])

class OctConv2dOut(OctConv2dBase):
	"""docstring for OctConv2dOut"""
	def __init__(self, in_channels, out_channels, in_alpha=0, ksize=None, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, *, dilate=1, groups=1):
		super(OctConv2dOut, self).__init__(in_channels, out_channels, in_alpha=in_alpha, out_alpha=0, ksize=ksize, stride=stride, pad=pad, nobias=nobias, initialW=initialW, initial_bias=initial_bias, dilate=dilate, groups=groups)
	
	def __call__(self, x):
		high, low = x

		high_to_high = self.conv_high_to_high(high)
		low_to_high = self.conv_low_to_high(F.avg_pooling_2d(low))

		return high_to_high + low_to_high
		