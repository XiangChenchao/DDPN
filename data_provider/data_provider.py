#coding:utf-8
import numpy as np
import re, json, random
from config.base_config import cfg
import sys
import os
import os.path as osp
import cv2
import skimage.io
from utils.bbox_transform import bbox_transform, clip_boxes, bbox_transform_inv
from utils.cython_bbox import bbox_overlaps
from utils.data_utils import save, load, transform_single
from utils.dictionary import Dictionary

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)


class DataProvider(object):
    def __init__(self, data_split, batchsize=1):
        print 'init DataProvider for %s : %s : %s' % (cfg.IMDB_NAME, cfg.PROJ_NAME, data_split)
        self.is_ss = cfg.FEAT_TYPE == 'ss'
        self.ss_box_dir = cfg.SS_BOX_DIR
        self.ss_feat_dir = cfg.SS_FEAT_DIR
        self.feat_type = cfg.FEAT_TYPE

        if 'refcoco' in cfg.IMDB_NAME or cfg.IMDB_NAME == 'refclef':
            self.is_mscoco_prefix = True
        else:
            self.is_mscoco_prefix = False
        self.use_kld = cfg.USE_KLD
        # self.mscoco_prefix = cfg.MSCOCO_PREFIX
        self.rpn_topn = cfg.RPN_TOPN
        if self.is_ss:
            self.bottomup_feat_dim = cfg.SS_FEAT_DIM
        else:
            self.bottomup_feat_dim = cfg.BOTTOMUP_FEAT_DIM
        self.query_maxlen = cfg.QUERY_MAXLEN
        # self.data_paths = cfg.DATA_PATHS
        self.image_ext = '.jpg'
        data_splits = data_split.split(cfg.SPLIT_TOK)
        if 'train' in data_splits:
            self.mode = 'train'
        else:
            self.mode = 'test'
        self.batchsize = batchsize
        self.image_dir = cfg.IMAGE_DIR
        self.feat_dir = cfg.FEAT_DIR
        self.dict_dir = osp.join(cfg.DATA_DIR, cfg.IMDB_NAME, 'query_dict')

        self.anno = self.load_data(data_splits)
        self.qdic = Dictionary(self.dict_dir)
        self.qdic.load()
        self.index = 0
        self.batch_len = None
        self.num_query = len(self.anno)


    def get_image_ids(self):
        qid_list = self.get_query_ids()
        iid_list = set()
        for qid in qid_list:
            iid_list.add(self.anno[qid]['iid'])
        return list(iid_list)

    def get_query_ids(self):
        return self.anno.keys()

    def get_num_query(self):
        return self.num_query

    def load_data(self, data_splits):
        anno = {}
        for data_split in data_splits:
            data_path = osp.join(cfg.DATA_DIR, cfg.IMDB_NAME, 'format_%s.pkl'%str(data_split))
            t_anno = load(data_path)
            anno.update(t_anno)
        return anno

    def get_vocabsize(self):
        return self.qdic.size()

    def get_iid(self, qid):
        return self.anno[qid]['iid']

    def get_img_path(self, iid):
        if self.is_mscoco_prefix:
            return os.path.join(self.image_dir, 'COCO_train2014_' + str(iid).zfill(12) + self.image_ext)
        else:
            return os.path.join(self.image_dir, str(iid) + self.image_ext)

    def str2list(self, qstr, query_maxlen):
        q_list = qstr.split()
        qvec = np.zeros(query_maxlen, dtype=np.int64)
        cvec = np.zeros(query_maxlen, dtype=np.int64)
        for i,_ in enumerate(xrange(query_maxlen)):
            if i < query_maxlen - len(q_list):
                cvec[i] = 0
            else:
                w = q_list[i-(query_maxlen-len(q_list))]
                # is the word in the vocabulary?
                # if self.qdic.has_token(w) is False:
                #     w = cfg.UNK_WORD    #'<unk>'
                qvec[i] = self.qdic.lookup(w)
                cvec[i] = 0 if i == query_maxlen - len(q_list) else 1

        return qvec, cvec

    def load_ss_box(self, ss_box_path):
        boxes = np.loadtxt(ss_box_path)
        if len(boxes) == 0:
            raise Exception("boxes is None!")
        boxes = boxes - 1
        boxes[:,[0,1]] = boxes[:,[1,0]]
        boxes[:,[2,3]] = boxes[:,[3,2]]
        return boxes

    def get_topdown_feat(self, iid):
        try:
            if self.is_ss:
                img_path = self.get_img_path(iid)
                im = skimage.io.imread(img_path)
                img_h = im.shape[0]
                img_w = im.shape[1]
                feat_path = os.path.join(self.ss_feat_dir, str(iid) + '.npz')
                ss_box_path = os.path.join(self.ss_box_dir, str(iid) + '.txt')
                bbox = self.load_ss_box(ss_box_path)
                num_bbox = bbox.shape[0]
                img_feat = np.transpose(np.load(feat_path)['x'], (1,0))
            else:
                if self.is_mscoco_prefix:
                    feat_path = os.path.join(self.feat_dir, 'COCO_train2014_' + str(iid).zfill(12) + self.image_ext + '.npz')
                else:
                    feat_path = os.path.join(self.feat_dir, str(iid) + self.image_ext + '.npz')
                feat_dict = np.load(feat_path)
                img_feat = feat_dict['x']
                num_bbox = feat_dict['num_bbox']
                bbox = feat_dict['bbox']
                img_h = feat_dict['image_h']
                img_w = feat_dict['image_w']
            return img_feat, num_bbox, bbox, (img_h, img_w)
        except Exception, e:
            print e
            raise Exception("UnkownError")

    def create_batch_rpn(self, iid):
        img_path = self.get_img_path(iid)
        # img = cv2.imread(img_path)
        img_feat, num_bbox, bbox, img_shape = self.get_topdown_feat(iid)
        return num_bbox, bbox, img_path

    def create_batch_recall(self, qid):
        iid = self.anno[qid]['iid']
        gt_bbox = self.anno[qid]['boxes']
        img_path = self.get_img_path(iid)
        # img = cv2.imread(img_path)
        img_feat, num_bbox, bbox, img_shape = self.get_topdown_feat(iid)
        return num_bbox, bbox, gt_bbox, img_path

    def compute_targets(self, ex_rois, gt_rois, query_label):
        """Compute bounding-box regression targets for an image."""
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 4

        targets = bbox_transform(ex_rois, gt_rois)
        if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - np.array(cfg.BBOX_NORMALIZE_MEANS))
                    / np.array(cfg.BBOX_NORMALIZE_STDS))
        query_bbox_target_data = np.hstack(
                (query_label[:, np.newaxis], targets)).astype(np.float32, copy=False)
        return query_bbox_target_data

    def get_query_bbox_regression_labels(self, query_bbox_target_data):
        query_label = query_bbox_target_data[:, 0]
        query_bbox_targets = np.zeros((query_label.size, 4), dtype=np.float32)
        query_bbox_inside_weights = np.zeros(query_bbox_targets.shape, dtype=np.float32)
        inds = np.where(query_label > 0)[0]
        if len(inds) != 0:
            for ind in inds:
                query_bbox_targets[ind, :] = query_bbox_target_data[ind, 1:]
                if query_label[ind] == 1:
                    query_bbox_inside_weights[ind, :] = cfg.BBOX_INSIDE_WEIGHTS
                elif query_label[ind] == 2:
                    query_bbox_inside_weights[ind, :] = 0.2

        return query_bbox_targets, query_bbox_inside_weights

    # 获取 query score和 bbox regression 的 label, mask
    def get_labels(self, rpn_rois, gt_boxes):
        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(rpn_rois, dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

        if self.use_kld:
            query_label = np.zeros(rpn_rois.shape[0])
        query_label_mask = 0
        bbox_label = np.zeros(rpn_rois.shape[0])
        # keep_inds = []
        #找出 query = 1 的 gt_box 的 index
        query_gt_ind = 0
        query_overlaps = overlaps[:,query_gt_ind].reshape(-1)

        if self.use_kld:
            # kld: 根据 iou 设置权重
            if query_overlaps.max() >= 0.5:
                query_label_mask = 1
                query_inds = np.where(query_overlaps>=cfg.THRESHOLD)[0]
                for ind in query_inds:
                    query_label[ind] = query_overlaps[ind]
                if query_label.sum() == 0:
                    print query_overlaps.max()
                query_label = query_label / float(query_label.sum())
        else:
            # softmax
            if query_overlaps.max() >= 0.5:
                query_label = int(query_overlaps.argmax())
            else:
                query_label = -1
        rois = rpn_rois
        gt_assignment = overlaps.argmax(axis=1)
        gt_target_boxes = gt_boxes[gt_assignment, :4]
        bbox_label[np.where(overlaps.max(axis=1)>=0.5)[0]] = 2
        if query_overlaps.max() >= 0.5:
            query_inds = np.where(query_overlaps>=cfg.THRESHOLD)[0]
            bbox_label[query_inds] = 1
            gt_target_boxes[query_inds] = gt_boxes[query_gt_ind, :4]

        bbox_target_data = self.compute_targets( rois, gt_target_boxes, bbox_label)
        query_bbox_targets, query_bbox_inside_weights =  self.get_query_bbox_regression_labels(bbox_target_data)
        query_bbox_outside_weights = np.array(query_bbox_inside_weights > 0).astype(np.float32)

        return query_label, query_label_mask, query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights

    def get_spt_feat(self, bbox, img_shape):
        spt_feat = np.zeros((bbox.shape[0], 5), dtype=np.float)

        spt_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        spt_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        spt_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        spt_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        spt_feat[:, 4] = (bbox[:, 2] - bbox[:, 0])*(bbox[:, 3] - bbox[:, 1]) / float(img_shape[0]*img_shape[1])
        return spt_feat

