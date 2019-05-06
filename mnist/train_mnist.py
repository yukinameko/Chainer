import chainer
import chainer.links as L
import chainer.datasets
from chainer import training
from chainer.training import extensions
from mnist import MNIST
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--out', default='mnist_result')
args = parser.parse_args()

train, test = chainer.datasets.get_mnist(ndim=3)
train_iter = chainer.iterators.SerialIterator(train, 32)
test_iter = chainer.iterators.SerialIterator(train, 32, repeat=False, shuffle=False)

model = MNIST()
model = L.Classifier(model)

if args.gpu >= 0:
	chainer.cuda.get_device_from_id(args.gpu).use()
	model.to_gpu()

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (10, 'epoch'), out=args.out)

trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
