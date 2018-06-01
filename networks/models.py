import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from config.base_config import cfg
import time, json
import gc


def net(split, vocab_size, opts):
    n = caffe.NetSpec()
    param_str = json.dumps({'split':split, 'batchsize':cfg.BATCHSIZE})
    n.qvec, n.cvec, n.img_feat, n.spt_feat, n.query_label, n.query_label_mask, n.query_bbox_targets, \
                n.query_bbox_inside_weights, n.query_bbox_outside_weights =  L.Python( \
                                        name='data', module='networks.data_layer', layer='DataProviderLayer', param_str=param_str, ntop=9 )

    n.embed_ba = L.Embed(n.qvec, input_dim=vocab_size, num_output=cfg.WORD_EMB_SIZE, \
                         weight_filler=dict(type='xavier'))
    n.embed = L.TanH(n.embed_ba)
    word_emb = n.embed

    # LSTM1
    n.lstm1 = L.LSTM(\
                   word_emb, n.cvec,\
                   recurrent_param=dict(\
                       num_output=cfg.RNN_DIM,\
                       weight_filler=dict(type='xavier')))

    tops1 = L.Slice(n.lstm1, ntop=cfg.QUERY_MAXLEN, slice_param={'axis':0})
    for i in xrange(cfg.QUERY_MAXLEN-1):
        n.__setattr__('slice_first'+str(i), tops1[int(i)])
        n.__setattr__('silence_data_first'+str(i), L.Silence(tops1[int(i)],ntop=0))
    n.lstm1_out = tops1[cfg.QUERY_MAXLEN-1]
    n.lstm1_reshaped = L.Reshape(n.lstm1_out, reshape_param=dict( shape=dict(dim=[-1,cfg.RNN_DIM])))
    n.lstm1_droped = L.Dropout(n.lstm1_reshaped,dropout_param={'dropout_ratio':cfg.DROPOUT_RATIO})
    n.lstm_l2norm = L.L2Normalize(n.lstm1_droped)
    n.q_emb = L.Reshape(n.lstm_l2norm, reshape_param=dict( shape=dict(dim=[0, -1])))
    q_layer = n.q_emb       # (N, 1024)


    v_layer = proc_img(n, n.img_feat, n.spt_feat)    #out: (N, 100, 2053)
    out_layer = concat(n, q_layer, v_layer)
    # predict score
    n.query_score_fc = L.InnerProduct(out_layer, num_output=1, weight_filler=dict(type='xavier'))
    n.query_score_pred = L.Reshape(n.query_score_fc, reshape_param=dict(shape=dict(dim=[-1, cfg.RPN_TOPN])))
    if cfg.USE_KLD:
        n.loss_query_score = L.SoftmaxKLDLoss(n.query_score_pred, n.query_label, n.query_label_mask, propagate_down=[1,0,0], loss_weight=1.0)
    else:
        n.loss_query_score = L.SoftmaxWithLoss(n.query_score_pred, n.query_label, n.query_label_mask, propagate_down=[1,0,0], loss_weight=1.0)

    # predict bbox
    n.query_bbox_pred = L.InnerProduct(out_layer, num_output=4, weight_filler=dict(type='xavier'))
    if cfg.USE_REG:
        n.loss_query_bbox = L.SmoothL1Loss( n.query_bbox_pred, n.query_bbox_targets, \
                                        n.query_bbox_inside_weights, n.query_bbox_outside_weights, loss_weight=1.0)
    else:
        n.__setattr__('silence_query_bbox_pred', L.Silence(n.query_bbox_pred, ntop=0))
        n.__setattr__('silence_query_bbox_targets', L.Silence(n.query_bbox_targets, ntop=0))
        n.__setattr__('silence_query_bbox_inside_weights', L.Silence(n.query_bbox_inside_weights, ntop=0))
        n.__setattr__('silence_query_bbox_outside_weights', L.Silence(n.query_bbox_outside_weights, ntop=0))
    return n.to_proto()


def proc_img(n, img_feat_layer, spt_feat_layer):
    # n.img_feat_resh = L.Reshape(img_feat_layer,reshape_param=dict(shape=dict(dim=[-1,cfg.BOTTOMUP_FEAT_DIM])))
    # n.spt_feat_resh = L.Reshape(spt_feat_layer,reshape_param=dict(shape=dict(dim=[-1,cfg.SPT_FEAT_DIM])))
    n.v_spt = L.Concat(img_feat_layer, spt_feat_layer, concat_param={'axis': 2})
    # if cfg.FUSE_TYPE == 'mfb' or cfg.FUSE_TYPE == 'mfh':
    #     n.v_emb1 = L.InnerProduct(n.v_spt, num_output=2048, 
    #                                    weight_filler=dict(type='xavier'))
    #     n.v_l2norm = L.L2Normalize(n.v_emb1)
    #     n.v_emb2 = L.InnerProduct(n.v_l2norm, num_output=cfg.JOINT_EMB_SIZE, 
    #                                    weight_filler=dict(type='xavier'))
    #     out_layer = n.v_emb2
    # elif cfg.FUSE_TYPE == 'concat':
    #     out_layer = n.v_spt
    out_layer = n.v_spt
    return out_layer

def concat(n, q_layer, v_layer):
    # input: q_layer:(N,1024)   v_layer:(N,100,2053)
    n.q_emb_resh1 = L.Reshape(q_layer,reshape_param=dict(shape=dict(dim=[0,1,cfg.RNN_DIM])))
    n.q_emb_tile = L.Tile(n.q_emb_resh1, axis=1, tiles=cfg.RPN_TOPN)
    n.q_emb_resh = L.Reshape(n.q_emb_tile,reshape_param=dict(shape=dict(dim=[-1,cfg.RNN_DIM])))

    n.v_emb_resh = L.Reshape(v_layer,reshape_param=dict(shape=dict(dim=[-1,cfg.SPT_FEAT_DIM+cfg.BOTTOMUP_FEAT_DIM])))
    n.qv_fuse = L.Concat(n.q_emb_resh, n.v_emb_resh, concat_param={'axis': 1})
    n.qv_fc1 = L.InnerProduct(n.qv_fuse, num_output=512, 
                                       weight_filler=dict(type='xavier'))
    n.qv_relu = L.ReLU(n.qv_fc1)
    return n.qv_relu


    





