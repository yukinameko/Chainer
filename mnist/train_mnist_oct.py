import chainer
import chainer.links as L
import chainer.datasets
from chainer import training
from chainer.training import extensions
from mnist_oct import MNIST_OCT

train, test = chainer.datasets.get_mnist(ndim=3)
train_iter = chainer.iterators.SerialIterator(train, 32)
test_iter = chainer.iterators.SerialIterator(train, 32, repeat=False, shuffle=False)

model = MNIST_OCT()
model = L.Classifier(model)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (10, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
