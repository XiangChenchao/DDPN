#coding:utf-8
import numpy as np
import os
import json
from utils.data_utils import save, load
from config.base_config import cfg


stop_words = [',', '<', '.', '>', '/', '?', '\'', '"', '\\', '-', '_', '=', '+', '[', ']', '{', '}', '|', ':', ';', '(', ')', '*', '&', '%', '^', '$', '#', '@', '!', '~', '`']

class Dictionary(object):
    """docstring for Dictionary"""
    def __init__(self, save_dir):
        self.idx2token = {}
        self.token2idx = {}
        self.word_freq = {}
        self.special = []
        self.save_dir = save_dir

    def save(self, save_dir = None):
        if not save_dir is None:
            self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save(self.idx2token, self.save_dir + '/idx2token.pkl')
        save(self.token2idx, self.save_dir + '/token2idx.pkl')
        save(self.word_freq, self.save_dir + '/word_freq.pkl')
        save(self.special, self.save_dir + '/special_words.pkl')

    def load(self):
        self.idx2token = load(self.save_dir + '/idx2token.pkl')
        self.token2idx = load(self.save_dir + '/token2idx.pkl')
        self.word_freq = load(self.save_dir + '/word_freq.pkl')
        self.special = load(self.save_dir + '/special_words.pkl')

    def split_words(self, s):
        s = s.strip().lower()
        for sw in stop_words:
            s = s.replace(sw, ' ')
        return s.split()

    def size(self):
        return len(self.token2idx)

    def add(self, token, idx=None):
        token = token.lower()
        if idx is None:
            idx = len(self.token2idx)

        if token not in self.token2idx:
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        idx = self.token2idx[token]
        if idx not in self.word_freq:
            self.word_freq[idx] = 1
        else:
            self.word_freq[idx] += 1
        return idx

    def get_tokens(self):
        return self.token2idx.keys()

    def add_tokens(self, tokens):
        vec = []
        for token in tokens:
            vec.append(self.add(token))
        return vec

    def add_specials(self, tokens, idxs):
        for idx, token in zip(idxs, tokens):
            self.add(token, idx)
        self.special += tokens

    def lookup(self, token, default=cfg.UNK):
        return self.token2idx.get(token, default)

    def has_token(self, token):
        return self.token2idx.has_key(token)

    def get_token(self, idx, default=cfg.UNK_WORD):
        return self.idx2token.get(idx, default)

    def merge(self, dic):
        self.add_tokens(dic.get_tokens())

    #convert sentence to index list
    def convert2idxs(self, tokens, unk_word):
        unk = self.lookup(unk_word)
        vec = [self.lookup(token, unk) for token in tokens]
        return vec

    def convert2tokens(self, idxs, unk_word):
        tokens = []
        tokens += [self.get_token(idx, unk_word) for idx in idxs]
        return tokens















