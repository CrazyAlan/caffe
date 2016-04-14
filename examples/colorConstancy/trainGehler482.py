#Copy data to local scratch
# import os
# datasetName = "geler"
# datasetFrom = "/home/xca64/remote/GitHub/colorP/dataSet/" + datasetName
# datasetTo = "/local-scratch/xca64/"
# cpCommand = "cp -rf " + datasetFrom + " " + datasetTo 
# rmCommand = "rm -rf " +  datasetTo + datasetName 
# print cpCommand 
# os.system(cpCommand)

import os
os.chdir('../../')
import sys
sys.path.insert(0, './python')
sys.path.append('/local-scratch/xca64/tmp/caffe-master/python/myFunc')
import caffe
import numpy as np
from pylab import *

caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('models/color_constancy/gehler_482_solver.prototxt')
solver.net.copy_from('models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

import tempfile

def run_solver(solver, niter, disp_interval, test_interval):
    blobs = ('loss', 'acc')
    loss, acc = (np.zeros(niter), np.zeros(niter))
    test_loss = (np.zeros(niter/test_interval)) 
    for it in range(niter):
        solver.step(1)  # run a single SGD step in Caffe
        loss[it] = (solver.net.blobs['loss'].data.copy())
        acc[it] = 0#(solver.net.blobs['loss_ang'].data.copy())
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = 'loss: %.3f'%loss[it]
            #print '%3d) %s Angular Erro %.3f' % (it, loss_disp, acc[it])     
            #print(solver.net.blobs['fc8_flickr'].data[1], solver.net.blobs['illu'].data[1])
        if it % test_interval == 0:
            print "%3d) test loss is %.3f"% (it ,solver.test_nets[0].blobs['accuracy'].data.copy())
            test_loss[it//test_interval] = solver.test_nets[0].blobs['accuracy'].data.copy()
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    name = 'firstTry'
    weights = {}
    filename = 'weights.%s.caffemodel' % name
    
    weights[name] = os.path.join(weight_dir, filename)
    solver.net.save(weights[name])
    
    return loss, test_loss, acc, weights


niters = 3000
loss_1, test_loss, acc_1, weights_1 = run_solver(solver, niters,20, 50)

print loss_1, test_loss


#solver.net.save('models/color_constancy/result/gehler_482_3000_iters.caffemodel')
#np.save('models/color_constancy/result/loss_1.npy', loss_1)