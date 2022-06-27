import tensorflow as tf
import cv2
import os
from os.path import join
import json
import random
import itertools
import numpy as np

abc = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']


def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(abc):
                outstr += abc[c]
        ret.append(outstr)
    return ret


def labels_to_text(labels):
    labels1 = []
    for i in labels:
        if i < len(abc):
            labels1.append(i)

    return ''.join(list(map(lambda x: abc[int(x)], labels1)))


def text_to_labels(text):
    a = list(map(lambda x: abc.index(x), text))
    return a


def is_valid_str(s):
    for ch in s:
        if not ch in abc:
            print('String is not valid.')
            return False
    return True


class TextImageGenerator:

    def __init__(self,
                 dirpath,
                 img_w, img_h,
                 batch_size,
                 downsample_factor,
                 max_text_len=9):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor

        img_dirpath = join(dirpath, 'img')
        ann_dirpath = join(dirpath, 'ann')
        self.samples = []

        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext == '.png':
                img_filepath = join(img_dirpath, filename)
                json_filepath = join(ann_dirpath, name + '.json')
                description = json.load(open(json_filepath, 'r'))['description']
                if is_valid_str(description) and len(description) > 0 and len(description) < 10:
                    self.samples.append([img_filepath, description])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0

    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
            bg = cv2.morphologyEx(img_gray, cv2.MORPH_DILATE, se)
            out_gray = cv2.divide(img_gray, bg, scale=255)
            out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
            img = cv2.resize(out_binary, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.texts.append(text)

    def get_output_size(self):
        return len(abc) + 1

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if tf.keras.backend.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len]) * 22
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if tf.keras.backend.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                aaaa = text_to_labels(text)
                laaaa = len(aaaa)
                Y_data[i, :laaaa] = aaaa
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield inputs, outputs
