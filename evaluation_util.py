import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
import cv2
import numpy as np
import torch
from skimage import transform as trans
from backbones import get_model
from sklearn.metrics import roc_curve, auc
import math
import heapq
from sklearn.model_selection import KFold

from prettytable import PrettyTable
from pathlib import Path

import sys


class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1, network='r50'):
        image_size = (112, 112)
        self.image_size = image_size
        weight = torch.load(prefix)
        resnet = get_model(network, dropout=0, fp16=False).cuda()
        resnet.load_state_dict(weight)
        model = torch.nn.DataParallel(resnet)
        self.model = model
        self.model.eval()
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg, landmark):
        if landmark is not None:
            assert landmark.shape[0] == 68 or landmark.shape[0] == 5
            assert landmark.shape[1] == 2
            if landmark.shape[0] == 68:
                landmark5 = np.zeros((5, 2), dtype=np.float32)
                landmark5[0] = (landmark[36] + landmark[39]) / 2
                landmark5[1] = (landmark[42] + landmark[45]) / 2
                landmark5[2] = landmark[30]
                landmark5[3] = landmark[48]
                landmark5[4] = landmark[54]
            else:
                landmark5 = landmark
            tform = trans.SimilarityTransform()
            tform.estimate(landmark5, self.src)
            M = tform.params[0:2, :]
            img = cv2.warpAffine(rimg,
                                M, (self.image_size[1], self.image_size[0]),
                                borderValue=0.0)
        else:
            if rimg.shape[:-1] == (112, 112):
                img = rimg
            else:
                img = cv2.resize(rimg, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()
        
        

def get_image_feature(img_path, img_names, landmarks, model_path, batch_size=32, already_align=False):
    '''
    Get the images features and align before if needed 
    
        Parameters: 
            img_path (string): path to the images directory
            img_names (list or array of string): the names of the images
            landmarks (dict): the dict that contain the mapping between the image names and the landmarks points
            model_path (string): The path to the pretrain model
            batch_size (int): the batch size to used for the model embedding
            already_align (bool): if True we don't align the images and we can put landmarks to None

        Returns: 
            img_feats (np array (n, 1024)): The image features of the images specify in paramaters (512) and the flipped images (512)
    '''
    data_shape = (3, 112, 112)

    print('files:', len(img_names))
    rare_size = len(img_names) % batch_size
    batch = 0
    img_feats = np.empty((len(img_names), 1024), dtype=np.float32)

    batch_data = np.empty((2 * batch_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, batch_size)
    for img_index, img_name in enumerate(img_names[:len(img_names) - rare_size]):
        if img_name[0] == "/":
            img = cv2.imread(img_path + img_name)
        else:
            img = cv2.imread(os.path.join(img_path, img_name))

        if not already_align:
            lmk = landmarks[img_name]
        else:
            lmk = None
            
        input_blob = embedding.get(img, lmk)

        batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
        batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
        if (img_index + 1) % batch_size == 0:
            print('batch', batch)
            img_feats[batch * batch_size:batch * batch_size +
                                         batch_size][:] = embedding.forward_db(batch_data)
            batch += 1

    batch_data = np.empty((2 * rare_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, rare_size)
    for img_index, img_name in enumerate(img_names[len(img_names) - rare_size:]):
        if img_name[0] == "/":
            img = cv2.imread(img_path + img_name)
        else:
            img = cv2.imread(os.path.join(img_path, img_name))
        
        if not already_align:
            lmk = landmarks[img_name]
        else:
            lmk = None
            
        input_blob = embedding.get(img, lmk)

        batch_data[2 * img_index][:] = input_blob[0]
        batch_data[2 * img_index + 1][:] = input_blob[1]
        if (img_index + 1) % rare_size == 0:
            print('batch', batch)
            img_feats[len(img_names) -
                      rare_size:][:] = embedding.forward_db(batch_data)
            batch += 1

    return img_feats
  
def load_probe_gallery_set(path_probe, path_gallery, delim=" "):
    '''
    Load the probe set and the gallery set from files specify in parameters
    
        Parameters: 
            path_probe (string): path to the probe set
            path_gallery (string): path to the gallery set
            delim (string): delimiter use in the file to separate the fields (",", " ")

        Returns: 
            probe_set (pandas df): the probe set that contains image name and image id
            gallery_set (pandas df): the gallery set that contains image name and image id
    '''
    probe_set = pd.read_csv(path_probe, sep=delim, header=None, names=["img_name", "img_id"])
    gallery_set = pd.read_csv(path_gallery, sep=delim, header=None, names=["img_name", "img_id"])
    return probe_set, gallery_set  
    
def get_map_names_to_id(img_names):
    '''
    Give an unique id to each image name
    
        Parameters: 
            img_names (list of string): all the image names used in the test

        Returns: 
            map_name_to_id (dict): map the image name to an unique id
    '''
    map_name_to_id = dict(zip(img_names, range(len(img_names))))
    return map_name_to_id

def convert_names_to_id(p, map_name_to_id):
    '''
    Map the images names to ids from the reference map build with all images
    
        Parameters: 
            p (list of string): the image names
            map_name_to_id: the dictionary build with get_map_names_to_id

        Returns: 
            p_ids: the corresponding ids of the images names
    '''
    p_ids = [map_name_to_id[e] for e in p]
    return p_ids
    
class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)
 
    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def calculate_accuracy(threshold, score, actual_issame):
    predict_issame = np.greater(score, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
 
    acc = float(tp + tn) / len(score)
    return acc

def compute_accuracy_with_best_threshold(scores, labels, nrof_folds=10):
    thresholds = np.arange(0, 1, 0.01)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(len(scores))

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        acc_train = np.zeros((len(thresholds)))
        for threshold_idx, threshold in enumerate(thresholds):
            acc_train[threshold_idx] = calculate_accuracy(
                threshold, scores[train_set], labels[train_set])

        best_threshold_index = np.argmax(acc_train)
        print("Best threshold for fold {0} is: {1}".format(fold_idx, thresholds[best_threshold_index]))
        accuracy[fold_idx] = calculate_accuracy(
                thresholds[best_threshold_index], scores[test_set],
                labels[test_set])

    acc, std = np.mean(accuracy), np.std(accuracy)
    print("Accuracy: {0} +/- {1}".format(acc, std))

    return acc, std

def verification(template_norm_feats=None,
                 img_names=None,
                 p1=None,
                 p2=None):
    '''
    Verification protocol (one-to-one)
    
        Parameters: 
            template_norm_feats (np array (n, 512)): the image features
            img_names (list of string): all the image names used
            p1 (list of string): the image names for the first pair
            p2 (list of string): the image names for the second pair

        Returns: 
            score (np array of float (# of pairs)): the score of the verification (cosine sim.)
    '''
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    map_name_to_id = get_map_names_to_id(img_names)
    p1 = convert_names_to_id(p1, map_name_to_id)
    p2 = convert_names_to_id(p2, map_name_to_id)

    p1 = np.array(p1)
    p2 = np.array(p2)

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[p1[s]]
        feat2 = template_norm_feats[p2[s]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score

def evaluation(query_feats, gallery_feats, mask, is_cmc=False):
    '''
    Evaluation protocol (one-to-many) and print top rank-n and cmc metric
    
        Parameters: 
            query_feats (np array (n, 512)): the image features of the gallery set
            gallery_feats (np array (n, 512)): the image features of the probe set
            mask (list of int): map identity from the probe to gallery set
            is_cmc (bool): if True use cmc metric
    '''
    Fars = [0.01, 0.1]
    print(query_feats.shape)
    print(gallery_feats.shape)

    query_num = query_feats.shape[0]
    gallery_num = gallery_feats.shape[0]

    similarity = np.dot(query_feats, gallery_feats.T)
    print('similarity shape', similarity.shape)
    top_inds = np.argsort(-similarity)
    print(top_inds.shape)

    # calculate top1
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0]
        if j == mask[i]:
            correct_num += 1
    print("top1 = {}".format(correct_num / query_num))
    # calculate top5
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:5]
        if mask[i] in j:
            correct_num += 1
    print("top5 = {}".format(correct_num / query_num))
    # calculate 10
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:10]
        if mask[i] in j:
            correct_num += 1
    print("top10 = {}".format(correct_num / query_num))

    if is_cmc:
        correct_nums = np.zeros(20)
        for i in range(query_num):
            for k in range(1, 21):
                j = top_inds[i, 0:k]
                if mask[i] in j:
                    correct_nums[k-1:] += 1
                    break
        cmc = correct_nums / query_num

        print("CMC (1-20): {0}".format(cmc))
        print_cmc(cmc, "RFW")

    neg_pair_num = query_num * gallery_num - query_num
    print(neg_pair_num)
    required_topk = [math.ceil(query_num * x) for x in Fars]
    top_sims = similarity
    # calculate fars and tprs
    pos_sims = []
    for i in range(query_num):
        gt = mask[i]
        pos_sims.append(top_sims[i, gt])
        top_sims[i, gt] = -2.0

    pos_sims = np.array(pos_sims)
    print(pos_sims.shape)
    neg_sims = top_sims[np.where(top_sims > -2.0)]
    print("neg_sims num = {}".format(len(neg_sims)))
    neg_sims = heapq.nlargest(max(required_topk), neg_sims)  # heap sort
    print("after sorting , neg_sims num = {}".format(len(neg_sims)))
    for far, pos in zip(Fars, required_topk):
        th = neg_sims[pos - 1]
        recall = np.sum(pos_sims > th) / query_num
        print("far = {:.10f} pr = {:.10f} th = {:.10f}".format(
            far, recall, th))


def print_roc(scores, label, target="dataset_name", method="method_name", save_path="."):
    '''
    Print the roc curve metric
    
        Parameters: 
            scores (np array of float (# of pairs)): the scores of the verification protocols
            label (list of int (1 or 0)): the corresponding label to each score
            target (string): name of the dataset
            method (string): name of the method
            save_path (string): path where to save the results
    '''
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
    fig = plt.figure()

    fpr, tpr, _ = roc_curve(label, scores)
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
                tpr,
                lw=1,
                label=('[(AUC = %0.4f %%)]' %
                    (roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
    plt.xlim([10 ** -6, 0.1])
    plt.ylim([0.3, 1])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels)
    plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on {}'.format(target))
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))
    print(tpr_fpr_table)

def print_cmc(cmc_array, target, save_path="."):
    '''
    Print the cmc curve metric
    
        Parameters: 
            cmc_array (list of float): the ranks accuracy from 1 to 20
            target (string): name of the dataset
            save_path (string): path where to save the results
    '''
    fig = plt.figure()
    plt.plot(range(1, 21), cmc_array)
    plt.ylim([0, 1])
    plt.xlabel('Rank')
    plt.ylabel('Accuracy')
    plt.title('CMC on {}'.format(target))
    fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))