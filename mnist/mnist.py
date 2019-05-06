import chainer
import chainer.functions as F
import chainer.links as L

class MNIST(chainer.Chain):
	"""docstring for MNIST"""
	def __init__(self, train=True):
		super(MNIST, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1)
			self.conv2 = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1)

			self.conv3 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
			self.conv4 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)

			self.conv5 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
			self.conv6 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)

			self.fc1 = L.Linear(None, 256)
			self.fc2 = L.Linear(None, 10)
		
	def __call__(self, x):
		h = F.relu(self.conv1(x))
		h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), ksize=2, stride=2)

		h = F.relu(self.conv3(h))
		h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv4(h))), ksize=2, stride=2)

		h = F.relu(self.conv5(h))
		h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv6(h))), ksize=2, stride=2)

		h = F.dropout(F.relu(self.fc1(h)))
		h = self.fc2(h)

		return h
