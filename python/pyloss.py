import caffe
import numpy as np
from colorConstancy import * 

class EuclideanLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num


class AngularLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        #self.diff[...] = bottom[0].data - bottom[1].data
        #print bottom[0].num
        top[0].data[...] = np.sum(multiangle(bottom[0].data, bottom[1].data)) / bottom[0].num/1.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                bottom[i].diff[...] = deMultiangle2(bottom[0].data, bottom[1].data)/bottom[0].num/1.
            else:
                bottom[i].diff[...] = deMultiangle2(bottom[1].data, bottom[0].data)/bottom[0].num/1.
            
class AngularAccLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 6:
            raise Exception("Need six inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        #self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        #self.diff[...] = bottom[0].data - bottom[1].data
        #print bottom[0].num
        a0 = bottom[0].data
        a1 = bottom[1].data
        a2 = bottom[2].data

        g0 = bottom[3].data
        g1 = bottom[4].data
        g2 = bottom[5].data

        approx = (np.array([a0, a1, a2]).transpose())
        actual = (np.array([g0, g1, g2]).transpose()) 
        #print bottom[0].num
        print approx

        #top[0].data[...] = np.sum(multiangle(approx, actual)) / bottom[0].num/1.

    def backward(self, top, propagate_down, bottom):
        pass
        # for i in range(2):
        #     if not propagate_down[i]:
        #         continue
        #     if i == 0:
        #         sign = 1
        #     else:
        #         sign = -1
        #     bottom[i].diff[...] = sign * self.diff / bottom[i].num


class SilenceLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        pass

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass