#coding:utf-8
import numpy as np
import re, json, random
from config.base_config import cfg
import sys
from data_provider import DataProvider
import torch.utils.data as data

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)



class MultiDataProvider(data.Dataset, DataProvider):
    """docstring for DataProvider"""
    def __init__(self, data_split, batchsize=1):
        DataProvider.__init__(self, data_split, batchsize)


    def __getitem__(self, index):
        if self.batch_len is None:
            self.n_skipped = 0
            qid_list = self.get_query_ids()
            if self.mode == 'train':
                random.shuffle(qid_list)
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.epoch_counter = 0
            print 'mode %s has %d data'%(self.mode, self.batch_len)

        qid = self.qid_list[index]
        gt_bbox = np.zeros(4)
        qvec = np.zeros(self.query_maxlen)
        cvec = np.zeros(self.query_maxlen)
        img_feat = np.zeros((self.rpn_topn, self.bottomup_feat_dim))
        bbox = np.zeros((self.rpn_topn, 4))
        img_shape = np.zeros(2)
        spt_feat = np.zeros((self.rpn_topn, 5))
        if self.use_kld:
            query_label = np.zeros(self.rpn_topn)
        # else:
        #     # softmax
        #     query_label = np.zeros(self.batchsize)
        # query_label_mask = np.zeros(self.batchsize)
        query_label_mask = 0
        query_bbox_targets = np.zeros((self.rpn_topn, 4))
        query_bbox_inside_weights = np.zeros((self.rpn_topn, 4))
        query_bbox_outside_weights = np.zeros((self.rpn_topn, 4))
        # valid_data = np.ones(self.batchsize)
        valid_data = 1

        t_qstr = self.anno[qid]['qstr']
        t_qvec, t_cvec = self.str2list(t_qstr, self.query_maxlen)
        qvec[...] = t_qvec
        cvec[...] = t_cvec

        try:
            t_gt_bbox = self.anno[qid]['boxes']
            gt_bbox[...] = t_gt_bbox[0]
            t_img_feat, t_num_bbox, t_bbox, t_img_shape = self.get_topdown_feat(self.anno[qid]['iid'])
            t_img_feat = t_img_feat.transpose((1, 0))
            t_img_feat = ( t_img_feat / np.sqrt((t_img_feat**2).sum()) )
            
            img_feat[:t_num_bbox, :] = t_img_feat
            bbox[:t_num_bbox, :] = t_bbox

            # spt feat
            img_shape[...] = np.array(t_img_shape)
            t_spt_feat = self.get_spt_feat(t_bbox, t_img_shape)
            spt_feat[:t_num_bbox, :] = t_spt_feat

            # query label, mask
            t_gt_bbox = np.array(self.anno[qid]['boxes'])
            t_query_label, t_query_label_mask, t_query_bbox_targets, t_query_bbox_inside_weights, t_query_bbox_outside_weights = \
                            self.get_labels(t_bbox, t_gt_bbox)
            if self.use_kld:
                query_label[:t_num_bbox] = t_query_label
                query_label_mask = t_query_label_mask
            else:
                query_label = t_query_label
            query_bbox_targets[:t_num_bbox, :] = t_query_bbox_targets
            query_bbox_inside_weights[:t_num_bbox, :] = t_query_bbox_inside_weights
            query_bbox_outside_weights[:t_num_bbox, :] = t_query_bbox_outside_weights

        except Exception, e:
            print e
            valid_data = 0
            if not self.use_kld:
                query_label = -1
            query_label_mask = 0
            query_bbox_inside_weights[...] = 0
            query_bbox_outside_weights[...] = 0
            print 'data not found for iid: %s'%str(self.anno[qid]['iid'])


        if self.index >= self.batch_len-1:
            self.epoch_counter += 1
            qid_list = self.get_query_ids()
            random.shuffle(qid_list)
            self.qid_list = qid_list
            print 'a epoch passed'

        return gt_bbox, qvec, cvec, img_feat, bbox, img_shape, spt_feat, query_label, query_label_mask, \
                                query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights, valid_data, int(self.anno[qid]['iid'])


    def __len__(self):
        return self.num_query
        









