#coding: utf-8
import caffe
import math
import numpy as np
import numpy.random as npr
import random
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform, clip_boxes, bbox_transform_inv
from utils.cython_bbox import bbox_overlaps

import yaml, cv2, os, sys
from utils.data_utils import save, load
from utils.dictionary import Dictionary



class FocalLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        layer_params = yaml.load(self.param_str_)
        self.gamma = cfg.GAMMA
        self.total = 0
        self.ignore = 0
        top[0].reshape(1)


    def forward(self, bottom, top):
        qrn_score = bottom[0].data.reshape(-1)
        rpn_rois = bottom[1].data[:,1:5]
        gt_boxes = bottom[2].data

        overlaps = bbox_overlaps(
            np.ascontiguousarray(rpn_rois[:, :4], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

        query_gt_ind = np.where(gt_boxes[:, 5] > 0)[0]
        query_overlaps = overlaps[:,query_gt_ind].reshape(-1)
        query_ind = query_overlaps.argmax()

        pt = qrn_score[query_ind]
        loss = - np.power(1 - pt, self.gamma) * np.log(pt)
        loss = np.array([loss])

        self.query_ind = query_ind
        self.qrn_score = qrn_score
        self.pt = pt

        top[0].reshape(*loss.shape)
        top[0].data[...] = loss


    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def backward(self, top, propagate_down, bottom):
        if not propagate_down[0]:
            return
        bottom[0].diff[0, :] = 0
        bottom[0].diff[0, self.query_ind] = self.gamma * np.power(1-self.pt, self.gamma-1) * \
                        np.log(self.pt) - np.power(1-self.pt, self.gamma) * 1.0 / float(self.pt)
        


