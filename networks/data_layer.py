import caffe
import numpy as np
import re, json, random, sys
from config.base_config import cfg
from data_provider.data_factory import get_data_provider
from utils.data_utils import complete_data
import functools

class DataProviderLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.bottomup_feat_dim = cfg.BOTTOMUP_FEAT_DIM
        self.query_maxlen = cfg.QUERY_MAXLEN
        self.split = json.loads(self.param_str_)['split']
        self.batchsize = json.loads(self.param_str_)['batchsize']
        self.use_kld = cfg.USE_KLD
        
        self.top_names = ['qvec','cvec','img_feat','spt_feat','query_label', 'query_label_mask', \
                                'query_bbox_targets', 'query_bbox_inside_weights', 'query_bbox_outside_weights']
        top[0].reshape(self.query_maxlen, self.batchsize)
        top[1].reshape(self.query_maxlen, self.batchsize)
        top[2].reshape(self.batchsize, cfg.RPN_TOPN,self.bottomup_feat_dim)
        top[3].reshape(self.batchsize, cfg.RPN_TOPN,5)
        if self.use_kld:
            top[4].reshape(self.batchsize, cfg.RPN_TOPN)
        else:
            top[4].reshape(self.batchsize)
        top[5].reshape(self.batchsize)
        top[6].reshape(self.batchsize*cfg.RPN_TOPN,4)
        top[7].reshape(self.batchsize*cfg.RPN_TOPN,4)
        top[8].reshape(self.batchsize*cfg.RPN_TOPN,4)

        if str(self.phase) == 'TRAIN':
            dp = get_data_provider(data_split=self.split, batchsize=self.batchsize)
            if cfg.NTHREADS > 1:
                import torch
                self.dataloader = torch.utils.data.DataLoader( dp,
                                                    batch_size=self.batchsize,
                                                    shuffle=True,
                                                    num_workers=int(cfg.NTHREADS))
            else:
                self.dataloader = dp

            self.data_iter = iter(self.dataloader)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        if str(self.phase) != 'TRAIN':
            return
        try:
            next_data = self.data_iter.next()
        except:
            self.data_iter = iter(self.dataloader)
            next_data = self.data_iter.next()

        next_data = map(np.array, next_data)
        my_complete_data = functools.partial(complete_data, batchsize=self.batchsize)
        gt_boxes, qvec, cvec, img_feat, bbox, img_shape, spt_feat, query_label, query_label_mask, \
                        query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights, valid_data, iid_list = map(my_complete_data, next_data)

        # queries
        qvec = np.transpose(qvec,(1,0))         # N x T -> T x N
        top[0].reshape(*qvec.shape)
        top[0].data[...] = qvec

        # query_cont
        cvec = np.transpose(cvec,(1,0))
        top[1].reshape(*cvec.shape)
        top[1].data[...] = cvec

        top[2].reshape(*img_feat.shape)
        top[2].data[...] = img_feat

        top[3].reshape(*spt_feat.shape)
        top[3].data[...] = spt_feat

        # query_label = query_label.reshape(-1)
        top[4].reshape(*query_label.shape)
        top[4].data[...] = query_label

        # query_label_mask
        top[5].reshape(*query_label_mask.shape)
        top[5].data[...] = query_label_mask

        # bbox regression
        query_bbox_targets = query_bbox_targets.reshape(-1, 4)
        top[6].reshape(*query_bbox_targets.shape)
        top[6].data[...] = query_bbox_targets

        query_bbox_inside_weights = query_bbox_inside_weights.reshape(-1, 4)
        top[7].reshape(*query_bbox_inside_weights.shape)
        top[7].data[...] = query_bbox_inside_weights

        query_bbox_outside_weights = query_bbox_outside_weights.reshape(-1, 4)
        top[8].reshape(*query_bbox_outside_weights.shape)
        top[8].data[...] = query_bbox_outside_weights


    def backward(self, top, propagate_down, bottom):
        pass
