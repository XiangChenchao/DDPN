#coding: utf-8
from config.base_config import cfg
from data_provider.data_factory import get_data_provider
from utils.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
import cPickle
from utils.cython_bbox import bbox_overlaps
from utils.dictionary import Dictionary
from utils.data_utils import complete_data
import os, sys, json
import functools
# import matplotlib.pyplot as plt


def calc_iou(box1, box2):
    x11 = min(box1[0], box1[2]); x12 = max(box1[0], box1[2])
    y11 = min(box1[1], box1[3]); y12 = max(box1[1], box1[3])
    x21 = min(box2[0], box2[2]); x22 = max(box2[0], box2[2])
    y21 = min(box2[1], box2[3]); y22 = max(box2[1], box2[3])
    # x11,y11,x12,y12 = box1
    # x21,y21,x22,y22 = box2
    x1 = max(x11, x21)
    x2 = min(x12, x22)
    y1 = max(y11, y21)
    y2 = min(y12, y22)
    if (x2 <= x1) or (y2 <= y1):
        return 0
        
    I = (x2-x1)*(y2-y1)
    if I <= 0:
        return 0
    U = (x12-x11)*(y12-y11) + (x22-x21)*(y22-y21) - I
    IOU = I / float(U)
    return IOU

def debug_rpn(debug_dir, iid, img, rois, num_rois=100):
    # debug
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    save_path = os.path.join(debug_dir, '%s.jpg'%str(iid))
    pred = img.copy()
    rois = rois[:num_rois]
    for box in rois:
        cv2.rectangle(pred, (box[0], box[1]), (box[2], box[3]), (0, 255,0))
    cv2.imwrite(save_path, pred)

# iid 相关
def test_net_rpn(test_split, topn_rois=cfg.TOPN_ROIS, threshold=cfg.OVERLAP_THRESHOLD, topk=cfg.TOPK, vis=False):
    dp = get_data_provider(test_split)
    iid_list = dp.get_image_ids()
    num_image = len(iid_list)

    for i, iid in enumerate(iid_list):
        num_bbox, bbox, img_path = dp.create_batch_rpn(iid)
        debug_dir = 'debug_rpn_%d'%topn_rois
        img = cv2.imread(img_path)
        img.shape
        debug_rpn(debug_dir, iid, img, bbox, topn_rois)

        percent = 100 * float(i) / num_image
        sys.stdout.write('\r' + ('%.2f' % percent) + '%')
        sys.stdout.flush()


def debug_recall(debug_dir, i, qvec, cvec, img, rois, num_rois=100):
    # debug
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    qdic_dir = osp.join(cfg.DATA_DIR, cfg.IMDB_NAME, 'query_dict')
    
    qdic = Dictionary(qdic_dir)
    qdic.load()
    cont = np.transpose(cvec)[0]
    query = np.transpose(qvec)[0]
    q_str = []
    for idx in query:
        if int(idx) != 0:
            q_str.append(qdic.get_token(idx))
    q_str = ' '.join(q_str)
    # if right_flag:
    #     save_dir = 'debug/%s/right/'%str(iid)+str(i)
    # else:
    #     save_dir = 'debug/%s/wrong/'%str(iid)+str(i)
    save_dir = os.path.join(debug_dir, '%s'%str(i))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # json.dump(list(cont.astype(np.int)), open(save_dir+'/%s'%q_str, 'w'))
    # json.dump(list(query.astype(np.int)), open(save_dir+'/query.json', 'w'))
    with open(save_dir+'/query.txt', 'w') as f:
        f.write(q_str)
    pred = img.copy()
    rois = rois[:num_rois]
    for box in rois:
        cv2.rectangle(pred, (box[0], box[1]), (box[2], box[3]), (0, 255,0))
    cv2.imwrite(save_dir+'/rois.jpg', pred)

#test ubp qid相关
def test_net_recall(test_split, threshold=cfg.OVERLAP_THRESHOLD, topk=cfg.TOPK, vis=False):
    dp = get_data_provider(test_split)
    qid_list = dp.get_query_ids()
    num_query = len(qid_list)
    num_right = 0
    pos_roi_num = 0
    y = [0 for i in range(1, 11)]
    x = [i*10 for i in range(1, 11)]


    for i, qid in enumerate(qid_list):
        # qvec, cvec, img_feat, rois, gt_bbox, img_path = dp.create_batch_recall(qid)
        num_bbox, rois, gt_bbox, img_path = dp.create_batch_recall(qid)
        # unscale back to raw image space
        # 计算前 10, 20, ...,100 个 proposal 的召回率
        boxes = rois
        for k, topn in enumerate(x):
            t_boxes = boxes[:topn, :]
            t_overlaps = bbox_overlaps(
                np.ascontiguousarray(t_boxes, dtype=np.float),
                np.ascontiguousarray(gt_bbox, dtype=np.float) )
            t_overlaps = t_overlaps[:,0].reshape(-1)
            if t_overlaps.max() >= 0.5:
                y[k] += 1

        overlaps = bbox_overlaps(
            np.ascontiguousarray(boxes, dtype=np.float),
            np.ascontiguousarray(gt_bbox, dtype=np.float) )
        overlaps = overlaps[:, 0].reshape(-1)
        if overlaps.max() >= 0.5:
            num_right += 1
        pos_roi_num += len(np.where(overlaps>0.5)[0])
        # tmp_num_right = 0
        # for k in range(boxes.shape[0]):
        #     iou = calc_iou(boxes[k], gt_bbox[0])
        #     if iou >= 0.5:
        #         tmp_num_right += 1
        # if tmp_num_right > 0:
        #     num_right += 1
        # pos_roi_num += tmp_num_right

        # debug rpn
        # debug_dir = 'debug_recall'
        # img = cv2.imread(img_path)
        # img.shape
        # debug_recall(debug_dir, i, qvec, cvec, img, boxes, cfg.TOPN_ROIS)

        percent = 100 * float(i) / num_query
        sys.stdout.write('\r' + ('%.2f' % percent) + '%')
        sys.stdout.flush()
    print 'upb: %f\n'%(num_right/float(num_query))
    print 'bpg: %f\n'%(pos_roi_num/float(num_query))


    y = [n/float(num_query) for n in y]
    # out_fig_path = 'recall.jpg'
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(x,y, color='blue')
    # ax1.set_xlabel('Iterations')
    # ax1.set_ylabel('Loss Value')
    # # plt.show()
    # plt.savefig(out_fig_path,dpi=200)
    # out_path = 'recall.json'
    # recall = {'x': x, 'y': y}
    # json.dump(recall, open(out_path, 'w'))
    return num_right/float(num_query), pos_roi_num/float(num_query)


def debug_pred(debug_dir, count, qvec, cvec, img, gt_bbox, roi, bbox_pred, iou):
    # debug
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    qdic_dir = osp.join(cfg.DATA_DIR, cfg.IMDB_NAME, 'query_dict')
    
    qdic = Dictionary(qdic_dir)
    qdic.load()
    q_str = []
    for idx in qvec:
        if int(idx) != 0:
            q_str.append(qdic.get_token(idx))
    q_str = ' '.join(q_str)
    if iou>=0.5:
        save_dir = os.path.join(debug_dir, 'right/'+str(count))
    else:
        save_dir = os.path.join(debug_dir, 'wrong/'+str(count))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir+'/%.3f'%iou, 'w') as f:
        f.write(' ')
    # json.dump(list(cont.astype(np.int)), open(save_dir+'/%s'%q_str, 'w'))
    # json.dump(list(query.astype(np.int)), open(save_dir+'/query.json', 'w'))
    with open(save_dir+'/query.txt', 'w') as f:
        f.write(q_str)
    pred = img.copy()
    box = gt_bbox.astype(np.int)
    cv2.rectangle(pred, (box[0], box[1]), (box[2], box[3]), (0, 255,0), 2)
    box = roi.astype(np.int)
    cv2.rectangle(pred, (box[0], box[1]), (box[2], box[3]), (0, 255,255), 2)
    box = bbox_pred.astype(np.int)
    cv2.rectangle(pred, (box[0], box[1]), (box[2], box[3]), (0, 0,255), 2)
    cv2.imwrite(save_dir+'/pred.jpg', pred)

def exec_validation(test_split, batchsize, gpu_id, prototxt, caffemodel, use_kld=cfg.USE_KLD, use_reg=cfg.USE_REG, threshold=cfg.OVERLAP_THRESHOLD, topk=cfg.TOPK, vis=False):
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    return test_net(test_split, net, batchsize, use_kld=use_kld, use_reg=use_reg, threshold=threshold, topk=topk, vis=vis)



def test_net(test_split, net, batchsize, use_kld=cfg.USE_KLD, use_reg=cfg.USE_REG, threshold=cfg.OVERLAP_THRESHOLD, topk=cfg.TOPK, vis=False):
    print 'validate split: %s'%test_split
    rpn_topn = cfg.RPN_TOPN
    dp = get_data_provider(data_split=test_split, batchsize=batchsize)
    num_query = dp.get_num_query()
    num_right = 0

    if cfg.NTHREADS > 1:
        try:
            import torch
            dataloader = torch.utils.data.DataLoader( dp,
                                        batch_size=batchsize,
                                        shuffle=False,
                                        num_workers=int(cfg.NTHREADS))
        except:
            cfg.NTHREADS = 1
            dataloader = dp
    else:
        dataloader = dp
    count = 0
    for data in dataloader:
        if data is None:
            break
        data = map(np.array, data)
        my_complete_data = functools.partial(complete_data, batchsize=batchsize)
        gt_boxes, qvec, cvec, img_feat, bbox, img_shape, spt_feat, query_label, query_label_mask, \
                        query_bbox_targets, query_bbox_inside_weights, query_bbox_outside_weights, valid_data, iid_list = map(my_complete_data, data)

        tp_qvec = qvec.copy()
        tp_cvec = cvec.copy()
        qvec = np.transpose(qvec,(1,0))
        cvec = np.transpose(cvec,(1,0))
        query_bbox_targets = query_bbox_targets.reshape(-1, 4)
        query_bbox_inside_weights = query_bbox_inside_weights.reshape(-1, 4)
        query_bbox_outside_weights = query_bbox_outside_weights.reshape(-1, 4)
        # net.blobs['queries'].reshape(*(qvec.shape))
        # net.blobs['query_cont'].reshape(*(cvec.shape))
        # net.blobs['img_feat'].reshape(*(img_feat.shape))
        # net.blobs['spt_feat'].reshape(*(spt_feat.shape))
        # net.blobs['query_label'].reshape(*query_label.shape)
        # net.blobs['query_label_mask'].reshape(*query_label_mask.shape)
        # net.blobs['query_bbox_targets'].reshape(*query_bbox_targets.shape)
        # net.blobs['query_bbox_inside_weights'].reshape(*query_bbox_inside_weights.shape)
        # net.blobs['query_bbox_outside_weights'].reshape(*query_bbox_outside_weights.shape)
        # forward_kwargs = {  'qvec': qvec.astype(np.float32, copy=False), \
        #                     'cvec': cvec.astype(np.float32, copy=False), \
        #                     'img_feat': img_feat.astype(np.float32, copy=False), \
        #                     'spt_feat': spt_feat.astype(np.float32, copy=False), \
        #                     'query_label': query_label.astype(np.float32, copy=False), \
        #                     'query_label_mask': query_label_mask.astype(np.float32, copy=False), \
        #                     'query_bbox_targets': query_bbox_targets.astype(np.float32, copy=False), \
        #                     'query_bbox_inside_weights': query_bbox_inside_weights.astype(np.float32, copy=False), \
        #                     'query_bbox_outside_weights': query_bbox_outside_weights.astype(np.float32, copy=False)}
        net.blobs['qvec'].data.reshape(*qvec.shape)
        net.blobs['qvec'].data[...] = qvec

        net.blobs['cvec'].data.reshape(*cvec.shape)
        net.blobs['cvec'].data[...] = cvec
        
        net.blobs['img_feat'].data.reshape(*img_feat.shape)
        net.blobs['img_feat'].data[...] = img_feat

        net.blobs['spt_feat'].data.reshape(*spt_feat.shape)
        net.blobs['spt_feat'].data[...] = spt_feat

        net.blobs['query_label'].data.reshape(*query_label.shape)
        net.blobs['query_label'].data[...] = query_label

        net.blobs['query_label_mask'].data.reshape(*query_label_mask.shape)
        net.blobs['query_label_mask'].data[...] = query_label_mask

        net.blobs['query_bbox_targets'].data.reshape(*query_bbox_targets.shape)
        net.blobs['query_bbox_targets'].data[...] = query_bbox_targets

        net.blobs['query_bbox_inside_weights'].data.reshape(*query_bbox_inside_weights.shape)
        net.blobs['query_bbox_inside_weights'].data[...] = query_bbox_inside_weights

        net.blobs['query_bbox_outside_weights'].data.reshape(*query_bbox_outside_weights.shape)
        net.blobs['query_bbox_outside_weights'].data[...] = query_bbox_outside_weights

        blobs_out = net.forward()
        # query_emb_tile = net.blobs['query_emb_tile'].data

        rois = bbox.copy()
        rois = rois.reshape(-1, 4)
        query_score_pred = net.blobs['query_score_pred'].data
        if use_reg:
            query_bbox_pred = net.blobs['query_bbox_pred'].data
            query_bbox_pred = bbox_transform_inv(rois, query_bbox_pred)
        else:
            query_bbox_pred = rois

        query_inds = np.argsort(-query_score_pred, axis=1)

        rois = rois.reshape(batchsize, rpn_topn, 4)
        query_bbox_pred = query_bbox_pred.reshape(batchsize, rpn_topn, 4)
        for i in range(batchsize):
            if valid_data[i] != 0:
                right_flag = False
                t_query_bbox_pred = clip_boxes(query_bbox_pred[i], img_shape[i])
                t_rois = clip_boxes(rois[i], img_shape[i])
                for j in range(topk):
                    query_ind = query_inds[i, j]
                    
                    # overlaps = bbox_overlaps(
                    #     np.ascontiguousarray(query_bbox_pred[query_ind][np.newaxis], dtype=np.float),
                    #     np.ascontiguousarray(gt_boxes, dtype=np.float) )
                    iou = calc_iou(t_query_bbox_pred[query_ind], gt_boxes[i])
                    # print '%.2f percent:  %.2f'%((100 * float(i) / num_query), 100*iou)
                    if iou >= threshold:
                        num_right += 1
                        right_flag = True
                        break
                    # if overlaps[0].max() > threshold:
                    #     # json.dump([1], open(save_dir + '/right.json', 'w'))
                    #     print overlaps[0].max()
                    #     num_right += 1
                    #     break
                
                # debug pred
                if vis:
                    debug_dir = 'visual_pred_%s_%s'%(cfg.IMDB_NAME, test_split)
                    img_path = dp.get_img_path(int(iid_list[i]))
                    img = cv2.imread(img_path)
                    img.shape
                    debug_pred(debug_dir, count, tp_qvec[i], tp_cvec[i], img, gt_boxes[i], t_rois[query_ind], t_query_bbox_pred[query_ind], iou)

            percent = 100 * float(count) / num_query
            sys.stdout.write('\r' + ('%.2f' % percent) + '%')
            sys.stdout.flush()
            count += 1
            if count >= num_query:
                break

    accuracy = num_right/float(num_query)
    print 'accuracy: %f\n'%accuracy
    return accuracy
    # with open('accuracy.txt', 'w') as f:
    #     f.write('accuracy: %f\n'%(num_right/float(num_query)))





















