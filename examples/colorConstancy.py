import os
os.chdir('..')
import sys
sys.path.insert(0, './python')

import caffe
import numpy as np
from pylab import *

from caffe import layers as L
from caffe import params as P

# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image


#Prototxt file path
proto_path = "models/finetune_flickr_style_2/train_val.prototxt"
weights = "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"

niter = 200
# losses will also be stored in the log
train_loss = np.zeros(niter)
scratch_train_loss = np.zeros(niter)

caffe.set_device(0)
caffe.set_mode_gpu()


dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
imagenet_net = caffe.Net(proto_path, weights, caffe.TEST)

imagenet_net.forward()

imgData = imagenet_net.blobs['data'].data.copy()

print(np.shape(imgData))

image = imgData[8,...]

plt.imshow(deprocess_net_image(image))
# We create a solver that fine-tunes from a previously trained network.
#solver = caffe.SGDSolver('models/finetune_flickr_style_2/solver.prototxt')
#solver.net.copy_from('models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
# For reference, we also create a solver that does no finetuning.
#scratch_solver = caffe.SGDSolver('models/finetune_flickr_style_2/solver.prototxt')



# We run the solver for niter times, and record the training loss.
# for it in range(niter):
#     solver.step(1)  # SGD by Caffe
#     scratch_solver.step(1)
#     # store the train loss
#     train_loss[it] = solver.net.blobs['loss'].data
#     scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data
#     if it % 10 == 0:
#         print 'iter %d, finetune_loss=%f, scratch_loss=%f' % (it, train_loss[it], scratch_train_loss[it])
# print 'done'

#Define Imagenet to classify








