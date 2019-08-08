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
import struct
import cv2

def read_bin(featFn):
    with open(featFn, "rb") as f:
        featLen, _, _, _ = struct.unpack("4i", f.read(16))
        return np.array(struct.unpack("%df" % (featLen), f.read()))

def calculate_sim(embeddings1, embeddings2):
    assert (len(embeddings1) == len(embeddings2))
    assert (len(embeddings1[0]) == len(embeddings2[0]))
    simList = []
    for idx in range(len(embeddings1)):
        '''
        l2_norm = cv2.norm(embeddings1[idx], cv2.NORM_L2)
        embeddings1_normal = embeddings1[idx]/ l2_norm
        l2_norm = cv2.norm(embeddings2[idx], cv2.NORM_L2)
        embeddings2_normal = embeddings2[idx]/ l2_norm
        score = np.dot(embeddings1_normal, embeddings2_normal.T)
        '''
        score = np.dot(embeddings1[idx], embeddings2[idx].T)
        simList.append(score)
    return np.array(simList)

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca = 0):
    assert(len(embeddings1) == len(embeddings2))
    assert(len(embeddings1[0]) == len(embeddings2[0]))
    nrof_pairs = min(len(actual_issame), len(embeddings1))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    #print('pca', pca)
    
    if pca==0:
        #diff = np.subtract(embeddings1, embeddings2)
        #dist = np.sum(np.square(diff),1)
        dist = calculate_sim(embeddings1, embeddings2)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #print('train_set', train_set)
        #print('test_set', test_set)
        if pca>0:
          print('doing pca on', fold_idx)
          embed1_train = embeddings1[train_set]
          embed2_train = embeddings2[train_set]
          _embed_train = np.concatenate( (embed1_train, embed2_train), axis=0 )
          #print(_embed_train.shape)
          pca_model = PCA(n_components=pca)
          pca_model.fit(_embed_train)
          embed1 = pca_model.transform(embeddings1)
          embed2 = pca_model.transform(embeddings2)
          embed1 = sklearn.preprocessing.normalize(embed1)
          embed2 = sklearn.preprocessing.normalize(embed2)
          #print(embed1.shape, embed2.shape)
          diff = np.subtract(embed1, embed2)
          dist = np.sum(np.square(diff),1)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        print('ROC: %d, best thresh: %f' % (fold_idx, thresholds[best_threshold_index]))
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    #predict_issame = np.less(dist, threshold)
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc
  
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (len(embeddings1) == len(embeddings2))
    assert (len(embeddings1[0]) == len(embeddings2[0]))
    nrof_pairs = min(len(actual_issame), len(embeddings1))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    #diff = np.subtract(embeddings1, embeddings2)
    #dist = np.sum(np.square(diff),1)
    dist = calculate_sim(embeddings1, embeddings2)
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        max_far_train = np.max(far_train)
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
        print('FAR: %d, best thresh: %f'%(fold_idx,threshold))
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_tar_far(thresholds, embeddings1, embeddings2, actual_issame, far_target):
    assert (len(embeddings1) == len(embeddings2))
    assert (len(embeddings1[0]) == len(embeddings2[0]))

    dist = calculate_sim(embeddings1, embeddings2)
    tar_total = np.zeros(len(thresholds))
    far_total = np.zeros(len(thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
        tar_total[threshold_idx], far_total[threshold_idx] = calculate_val_far(threshold, dist, actual_issame)

    if np.max(far_total) >= far_target:
        f = interpolate.interp1d(far_total, thresholds, kind='slinear')
        threshold = f(far_target)
    else:
        threshold = 0.0

    tar, far = calculate_val_far(threshold, dist, actual_issame)
    print('TAR: %f @ FAR: %f, thresh: %f' % (tar, far, threshold))


def calculate_val_far(threshold, dist, actual_issame):
    #predict_issame = np.less(dist, threshold)
    predict_issame = np.greater(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def evaluate(embeddings1, embeddings2, actual_issame, nrof_folds=10, pca = 0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, pca = pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)

    calculate_tar_far(thresholds, embeddings1, embeddings2, actual_issame, 1e-2)

    return tpr, fpr, accuracy, val, val_std, far

def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list1 = []
    path_list2 = []
    issame_list = []
    for pair in pairs:
        param = pair.split(',')
        path1 = lfw_dir + '/' + param[0] + file_ext
        path2 = lfw_dir + '/' + param[1] + file_ext
        if int(param[2]) == 1:
            issame = True
        else:
            issame = False
        if os.path.exists(path1) and os.path.exists(path2):    # Only add the pair if both paths exist
            path_list1.append(path1)
            path_list2.append(path2)
            issame_list.append(issame)
        else:
            print('not exists', path1, path2)
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    return path_list1, path_list2, issame_list

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        totalLines = f.readlines()
        for line in totalLines:
            pairs.append(line.strip())
    return pairs

def useFeatTestModel(lfw_feat_set):
    lfw_pairs = read_pairs(os.path.join(lfw_feat_set, 'pairs.txt'))
    feat_list1, feat_list2, issame_list = get_paths(lfw_feat_set, lfw_pairs, '.dat')
    print('testing lfw..')
    embeddings1 = []
    embeddings2 = []
    for i in xrange( len(feat_list1) ):
        featFn1 = feat_list1[i]
        feat1 = read_bin(featFn1)
        embeddings1.append(feat1)
        featFn2 = feat_list2[i]
        feat2 = read_bin(featFn2)
        embeddings2.append(feat2)

    #embeddings = embeddings_list[0].copy()
    #embeddings = sklearn.preprocessing.normalize(embeddings)
    _, _, accuracy, val, val_std, far = evaluate(embeddings1, embeddings2, issame_list, nrof_folds=10)
    acc1, std1 = np.mean(accuracy), np.std(accuracy)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    #embeddings = np.concatenate(embeddings_list, axis=1)
    #embeddings = embeddings_list[0] + embeddings_list[1]
    #embeddings = sklearn.preprocessing.normalize(embeddings)
    #print(embeddings.shape)
    #_, _, accuracy, val, val_std, far = evaluate(embeddings1, embeddings2, issame_list, nrof_folds=10)
    #acc2, std2 = np.mean(accuracy), np.std(accuracy)
    #return

if __name__ == "__main__":
    lfw_feat_set = 'E:/work/data/faceRecognize/faceRecEval/feats/lfw/mobileFaceNet'
    #lfw_feat_set = 'E:/work/data/faceRecognize/faceRecEval/feats/lfw/model-r34-amf'
    useFeatTestModel(lfw_feat_set)