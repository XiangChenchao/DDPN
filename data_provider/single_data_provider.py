#coding:utf-8
import numpy as np
import re, json, random
import sys
from data_provider import DataProvider


default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)



class SingleDataProvider(DataProvider):
    """docstring for DataProvider"""
    def __init__(self, data_split, batchsize=1):
        DataProvider.__init__(self, data_split, batchsize)


    def create_batch(self, qid_list):
        gt_bbox = np.zeros((self.batchsize, 4))
        qvec = (np.zeros(self.batchsize*self.query_maxlen)).reshape(self.batchsize,self.query_maxlen)
        cvec = (np.zeros(self.batchsize*self.query_maxlen)).reshape(self.batchsize,self.query_maxlen)
        img_feat = np.zeros((self.batchsize, self.rpn_topn, self.bottomup_feat_dim))
        bbox = np.zeros((self.batchsize, self.rpn_topn, 4))
        img_shape = np.zeros((self.batchsize, 2))
        spt_feat = np.zeros((self.batchsize, self.rpn_topn, 5))
        if self.use_kld:
            query_label = np.zeros((self.batchsize, self.rpn_topn))
        else:
            # softmax
            query_label = np.zeros(self.batchsize)
        query_label_mask = np.zeros((self.batchsize))
        query_bbox_targets = np.zeros((self.batchsize, self.rpn_topn, 4))
        query_bbox_inside_weights = np.zeros((self.batchsize, self.rpn_topn, 4))
        query_bbox_outside_weights = np.zeros((self.batchsize, self.rpn_topn, 4))
        valid_data = np.ones(self.batchsize)

        for i, qid in enumerate(qid_list):
            t_qstr = self.anno[qid]['qstr']
            t_qvec, t_cvec = self.str2list(t_qstr, self.query_maxlen)
            qvec[i, ...] = t_qvec
            cvec[i, ...] = t_cvec

            try:
                t_gt_bbox = self.anno[qid]['boxes']
                gt_bbox[i, ...] = t_gt_bbox[0]
                t_img_feat, t_num_bbox, t_bbox, t_img_shape = self.get_topdown_feat(self.anno[qid]['iid'])
                t_img_feat = t_img_feat.transpose((1, 0))
                t_img_feat = ( t_img_feat / np.sqrt((t_img_feat**2).sum()) )
                
                img_feat[i, :t_num_bbox, :] = t_img_feat
                bbox[i, :t_num_bbox, :] = t_bbox

                # spt feat
                img_shape[i, :] = np.array(t_img_shape)
                t_spt_feat = self.get_spt_feat(t_bbox, t_img_shape)
                spt_feat[i, :t_num_bbox, :] = t_spt_feat

                # query label, mask
                t_gt_bbox = np.array(self.anno[qid]['boxes'])
                t_query_label, t_query_label_mask, t_query_bbox_targets, t_query_bbox_inside_weights, t_query_bbox_outside_weights = \
                                self.get_labels(t_bbox, t_gt_bbox)
                if self.use_kld:
                    query_label[i, :t_num_bbox] = t_query_label
                    query_label_mask[i] = t_query_label_mask
                else:
                    query_label[i, ...] = t_query_label
                query_bbox_targets[i, :t_num_bbox, :] = t_query_bbox_targets
                query_bbox_inside_weights[i, :t_num_bbox, :] = t_query_bbox_inside_weights
                query_bbox_outside_weights[i, :t_num_bbox, :] = t_query_bbox_outside_weights

            except Exception, e:
                print e
                valid_data[i] = 0
                if not self.use_kld:
                    query_label[i] = -1
                query_label_mask[i] = 0
                query_bbox_inside_weights[i, ...] = 0
                query_bbox_outside_weights[i, ...] = 0
                print 'data not found for iid: %s'%str(self.anno[qid]['iid'])

        return gt_bbox, qvec, cvec, img_feat, bbox, img_shape, spt_feat, query_label, query_label_mask, \
                        query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights, valid_data

    def __getitem__(self, index):
        if self.batch_len is None:
            self.n_skipped = 0
            qid_list = self.get_query_ids()
            if self.mode == 'train':
                random.shuffle(qid_list)
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.batch_index = 0
            self.epoch_counter = 0
            print 'mode %s has %d data'%(self.mode, self.batch_len)

        counter = 0
        t_qid_list = []
        t_iid_list = []
        while counter < self.batchsize:
            t_qid = self.qid_list[self.batch_index]
            t_qid_list.append(t_qid)
            t_iid_list.append(self.get_iid(t_qid))
            counter += 1
            if self.batch_index < self.batch_len-1:
                self.batch_index += 1
            else:
                self.epoch_counter += 1
                qid_list = self.get_query_ids()
                random.shuffle(qid_list)
                self.qid_list = qid_list
                self.batch_index = 0
                print 'a epoch passed'
        t_batch = self.create_batch(t_qid_list)
        return t_batch + (t_iid_list,)

    def __len__(self):
        return self.num_query




