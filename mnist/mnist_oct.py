import sys, os
sys.path.append(os.pardir)
import chainer
import chainer.functions as F
import chainer.links as L

from OctaveConvolution.OctConv import OctConv2d, OctConv2dIn, OctConv2dOut

def relu_hl(x):
	h, l = x
	h = F.relu(h)
	l = F.relu(l)
	return [h, l]

class MNIST_OCT(chainer.Chain):
	"""docstring for MNIST"""
	def __init__(self, train=True):
		super(MNIST_OCT, self).__init__()
		with self.init_scope():
			self.conv1 = OctConv2dIn(None, 16, out_alpha=0.25, ksize=3, stride=1, pad=1)
			self.conv2 = OctConv2d(None, 16, out_alpha=0.25, ksize=3, stride=1, pad=1)

			self.conv3 = OctConv2d(None, 32, out_alpha=0.25, ksize=3, stride=1, pad=1)
			self.conv4 = OctConv2d(None, 32, out_alpha=0.25, ksize=3, stride=1, pad=1)

			self.conv5 = OctConv2dOut(None, 64, ksize=3, stride=1, pad=1)
			self.conv6 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)

			self.fc1 = L.Linear(None, 256)
			self.fc2 = L.Linear(None, 10)
		
	def __call__(self, x):
		h = relu_hl(self.conv1(x))
		h, l = self.conv2(h)
		h = F.max_pooling_2d(F.local_response_normalization(F.relu(h)), ksize=2, stride=2)
		l = F.max_pooling_2d(F.local_response_normalization(F.relu(l)), ksize=2, stride=2)

		h = relu_hl(self.conv3([h, l]))
		h, l = self.conv4(h)
		h = F.max_pooling_2d(F.local_response_normalization(F.relu(h)), ksize=2, stride=2)
		l = F.max_pooling_2d(F.local_response_normalization(F.relu(l)), ksize=2, stride=2)

		h = F.relu(self.conv5([h, l]))
		h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv6(h))), ksize=2, stride=2)

		h = F.dropout(F.relu(self.fc1(h)))
		h = self.fc2(h)

		return h
