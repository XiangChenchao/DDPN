#import tensorflow as tf
import numpy as np
# import scipy.misc 
import os, sys
# from PIL import Image



class Logger(object):
    
    def __init__(self, imdb_name, feat_type, proj_name, log_dir):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        self.feat_type = feat_type
        self.imdb_name = imdb_name
        self.proj_name = proj_name
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        #self.writer = tf.summary.FileWriter(self.log_dir)
        self.text_logger_path = os.path.join(self.log_dir, '%s_%s_%s_accuracy.txt'%(self.imdb_name, self.feat_type, self.proj_name))

    def scalar_summary(self, value, step):
        """Log a scalar variable."""
        #summary = tf.Summary(value=[tf.Summary.Value(tag='%s_%s_%s'%(self.imdb_name, self.feat_type, self.proj_name), simple_value=value)])
        #self.writer.add_summary(summary, step)
        with open(self.text_logger_path, 'a') as f:
            f.write('step: %s, acc: %f\n'%(str(step), value))

