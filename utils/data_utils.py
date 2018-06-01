#coding:utf-8
import numpy as np
import cv2
import cPickle
import os, sys
import time

def save(data, save_path):
    with open(save_path, 'wb') as f:
        cPickle.dump(data, f)

def load(file_path):
    with open(file_path, 'rb') as f:
        return cPickle.load(f)

def progress_bar(idx, total, last_time):
    elapse = time.time() - last_time
    percent = 100 * float(idx) / total
    sys.stdout.write('\r' + ('---elapse: %d seconds-----%.2f' % (elapse, percent)) + '%')
    sys.stdout.flush()
    return time.time()

#change x,y,w,h => x1,y1,x2,y2
def transform_inv(bbox):
    res = np.zeros_like(bbox)
    x = bbox[:,0]
    y = bbox[:,1]
    w = bbox[:,2]
    h = bbox[:,3]
    x1 = x - w / float(2)
    x2 = x + w / float(2)
    y1 = y - h / float(2)
    y2 = y + h / float(2)
    res = np.hstack((x1, y1, x2, y2))
    return res

#change x1,y1,x2,y2 => x,y,w,h
def transform_single(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    x = (x1 + x2)/ float(2)
    y = (y1 + y2)/ float(2)
    w = x2 - x1
    h = y2 - y1
    return np.hstack((x,y,w,h))


def complete_data(data, batchsize):
    if data.shape[0] == batchsize:
        return data
    if len(data.shape) == 1:
        t_data = np.zeros(batchsize - data.shape[0])
        return np.hstack((data, t_data))
    else:
        shape = (batchsize - data.shape[0] ,) + data.shape[1:]
        t_data = np.zeros(shape)
        return np.vstack((data, t_data))