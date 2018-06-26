import caffe
from config.base_config import cfg, get_models_dir
from utils.timer import Timer
import numpy as np
import os, sys, re
import cPickle
import os.path as osp

from caffe.proto import caffe_pb2
import google.protobuf as pb2
from test import exec_validation
from utils.logger import Logger

iter_reg = re.compile(r'(\d+)')

def add_bbox_regression_targets():
    # Use fixed / precomputed "means" and "stds" instead of empirical values
    query_means = np.array(cfg.BBOX_NORMALIZE_MEANS)
    query_stds = np.array(cfg.BBOX_NORMALIZE_STDS)
    return query_means, query_stds

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_path, opts):
        self.opts = opts
        self.models_dir = get_models_dir()
        self.pretrained_model = opts.pretrained_model
        self.snapshot_iters = cfg.TRAIN.SNAPSHOT_ITERS
        self.validate_interval = cfg.TRAIN.VALIDATE_INTERVAL
        self.val_net_path = opts.val_net_path
        """Initialize the SolverWrapper."""
        self.logger = Logger(cfg.IMDB_NAME, cfg.FEAT_TYPE, cfg.PROJ_NAME, cfg.LOG_DIR)

        self.niter = 0
        # self.solver = caffe.SGDSolver(solver_path)
        self.solver = caffe.get_solver(solver_path)
        if self.pretrained_model is not None and osp.exists(self.pretrained_model):
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.solver.net.copy_from(self.pretrained_model)
            modelname = self.pretrained_model.split('/')[-1]
            niter = iter_reg.search(modelname).group(1)
            try:
                niter = int(niter)
                self.solver.iter = niter
                self.niter = niter
            except:
                self.niter = 0

        if cfg.USE_REG:
            self.bbox_pred_layer_name = 'query_bbox_pred'
            print 'Computing bounding-box regression targets...'
            self.query_means, self.query_stds = add_bbox_regression_targets()
            print 'done'
            found = False
            for k in self.solver.net.params.keys():
                if self.bbox_pred_layer_name in k:
                    bbox_pred = k
                    found = True
                    print('[#] Renormalizing the final layers back')
                    self.solver.net.params[bbox_pred][0].data[...] = \
                        (self.solver.net.params[bbox_pred][0].data *
                         1.0 / self.query_stds[:,np.newaxis])
                    self.solver.net.params[bbox_pred][1].data[...] = \
                            (self.solver.net.params[bbox_pred][1].data - self.query_means) * 1.0 / self.query_stds
            if not found:
                print('Warning layer \"bbox_pred\" not found')


    # notice: for some reason we do not save learning rate
    def snapshot(self, filename=None):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        if cfg.USE_REG:
            # # save original values
            # orig_0 = net.params['bbox_pred'][0].data.copy()
            # orig_1 = net.params['bbox_pred'][1].data.copy()

            # # scale and shift with bbox reg unnormalization; then save snapshot
            # net.params['bbox_pred'][0].data[...] = \
            #         (net.params['bbox_pred'][0].data *
            #          self.bbox_stds[:, np.newaxis])
            # net.params['bbox_pred'][1].data[...] = \
            #         (net.params['bbox_pred'][1].data *
            #          self.bbox_stds + self.bbox_means)


            # save original values
            q_orig_0 = net.params[self.bbox_pred_layer_name][0].data.copy()
            q_orig_1 = net.params[self.bbox_pred_layer_name][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params[self.bbox_pred_layer_name][0].data[...] = \
                    (net.params[self.bbox_pred_layer_name][0].data *
                     self.query_stds[:,np.newaxis])
            net.params[self.bbox_pred_layer_name][1].data[...] = \
                    (net.params[self.bbox_pred_layer_name][1].data *
                     self.query_stds + self.query_means)


        if filename is None:
            infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                     if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
            filename = os.path.join(self.models_dir, infix +
                        '_iter_{:d}'.format(self.niter) + '.caffemodel')
            # filename = os.path.join(self.output_dir, filename)
        else:
            filename = os.path.join(self.models_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.USE_REG:
            # # restore net to original state
            # net.params['bbox_pred'][0].data[...] = orig_0
            # net.params['bbox_pred'][1].data[...] = orig_1

            # restore net to original state
            net.params[self.bbox_pred_layer_name][0].data[...] = q_orig_0
            net.params[self.bbox_pred_layer_name][1].data[...] = q_orig_1
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.niter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            self.niter += 1
            timer.toc()
            if self.niter % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.niter % self.snapshot_iters == 0:
                last_snapshot_iter = self.niter
                model_paths.append(self.snapshot())

            if self.niter % self.validate_interval == 0:
                self.snapshot('tmp.caffemodel')
                caffemodel = os.path.join(self.models_dir, 'tmp.caffemodel')
                accuracy = exec_validation(self.opts.val_split, cfg.BATCHSIZE, self.opts.gpu_id, self.val_net_path, caffemodel, \
                                            use_kld=cfg.USE_KLD, use_reg=cfg.USE_REG, threshold=cfg.OVERLAP_THRESHOLD, topk=cfg.TOPK, vis=self.opts.vis_pred)
                self.logger.scalar_summary(accuracy, self.niter)
        if last_snapshot_iter != self.niter:
            model_paths.append(self.snapshot())
        
def train_net(solver_path, opts):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_path, opts)

    print 'Solving...'
    sw.train_model(cfg.TRAIN.MAX_ITERS)
    print 'done solving'
