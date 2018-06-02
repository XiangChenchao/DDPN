import _init_path

from test import test_net
from config.base_config import cfg, print_cfg, get_models_dir, cfg_from_file
import caffe
from networks import models
import argparse
import pprint
import time, os, sys 
import numpy as np
import os.path as osp
from utils.dictionary import Dictionary


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Visual Grounding network')
    parser.add_argument('--gpu_id', help='gpu_id', default=0, type=int)
    parser.add_argument('--test_split', help='test_split', default='val', type=str)
    parser.add_argument('--batchsize', help='batchsize', default=64, type=int)
    parser.add_argument('--vis_pred', help='visualize prediction', default=False, type=bool)
    parser.add_argument('--test_net', help='test_net prototxt', 
                        default=None,
                        type=str)
    parser.add_argument('--pretrained_model', help='pretrained_model', 
                        type=str)
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        #default='config/experiments/refcoco-kld-bbox_reg.yaml',
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = parse_args()

    # print('Using config:')
    # pprint.pprint(cfg)
    
    if opts.cfg_file is not None:
        cfg_from_file(opts.cfg_file)
    print_cfg()

    if opts.test_net is None:
        qdic_dir = osp.join(cfg.DATA_DIR, cfg.IMDB_NAME, 'query_dict')
        qdic = Dictionary(qdic_dir)
        qdic.load()
        vocab_size = qdic.size()
        test_model = models.net(opts.test_split, vocab_size, opts)
        test_net_path = osp.join(get_models_dir(), 'test.prototxt')
        with open(test_net_path, 'w') as f:
            f.write(str(test_model))
    else:
        test_net_path = opts.test_net

    caffe.set_mode_gpu()
    caffe.set_device(opts.gpu_id)
    net = caffe.Net(test_net_path, opts.pretrained_model, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(opts.pretrained_model))[0]

    log_file = osp.join(cfg.LOG_DIR, '%s_%s_%s_accuracy.txt'%(cfg.IMDB_NAME, cfg.FEAT_TYPE, cfg.PROJ_NAME))
    if os.path.exists(log_file):
        os.remove(log_file)
    test_split = opts.test_split
    if type(test_split) is list:
        for split in test_split:
            accuracy = test_net(split, net, opts.batchsize, vis=opts.vis_pred)
            with open(log_file, 'a') as f:
                f.write('%s accuracy: %f\n'%(split, accuracy))
    else:
        accuracy = test_net(test_split, net, opts.batchsize, vis=opts.vis_pred)
        with open(log_file, 'a') as f:
            f.write('%s accuracy: %f\n'%(test_split, accuracy))
    
