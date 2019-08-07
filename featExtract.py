#coding=utf-8
"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import cv2
import sys
import numpy as np
import struct

def load_file_list(anno_fn):
    list = []
    with open(anno_fn, 'r') as f_anno:
        while True:
            line = f_anno.readline()
            if not line:
                break
            file_name = line.strip()
            list.append(file_name)
    return list

def write_bin(feats, featFn):
    feature = list(feats)
    with open(featFn, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))

def get_feature(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # color transform:BGR---RGB
    img = np.transpose(img, (2, 0, 1))
    input_blob = np.expand_dims(img, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()[0]
    l2_norm = cv2.norm(embedding, cv2.NORM_L2)
    return embedding / l2_norm

ctx = mx.cpu()
image_size = (112, 112)
#prefix = './models/face/model'
prefix = 'E:/work/github/insightface/insightface/models/model-r34-amf/model'
lfw_feat_dir = 'E:/work/data/faceRecognize/faceRecEval/feats/lfw/model-r34-amf/'

epoch = 0
print('loading',prefix, epoch)
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
all_layers = sym.get_internals()
sym = all_layers[ 'fc1' + '_output']
model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
#model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
model.set_params(arg_params, aux_params)


lfw_img_dir = 'E:/work/data/faceRecognize/insightface_test/lfw/'


for idx in range(12001):
    imgFn = lfw_img_dir + str(idx) + '.jpg'
    if not os.path.exists(imgFn):
        continue
    print('idx: %d, line: %s' % (idx, imgFn))
    img = cv2.imread(imgFn)
    feat = get_feature(model, img)
    featFn = lfw_feat_dir + str(idx) + '.dat'
    write_bin(feat, featFn)

