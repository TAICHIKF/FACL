'''
Descripttion: 
version: 
Author: TAICHIFEI
Date: 2022-07-04 11:29:37
LastEditors: TAICHIFEI
LastEditTime: 2022-08-01 20:54:46
'''
import os
import torch
import pandas as pd
from tqdm import tqdm
from collections import  OrderedDict

# from utils import read_yaml
import models
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import metrics

from models.model_attention_mil import MIL_Attention_fc
from utils.utils import *
import yaml
from addict import Dict

def read_yaml(fpath="./configs/sample.yaml"):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)



def load_model_(cfg, ckpt_path=None, fold=0):
    print('Init Model {}'.format(fold))    
    model_dict = {'n_classes': cfg.General.n_classes}
    
    model = MIL_Attention_fc(**model_dict) 

    # print_network(model)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_clean = {}
        for key in ckpt.keys():
            ckpt_clean.update({key.replace('attention_net.3.attention_', 'attention_net.2.attention_'):ckpt[key]})

        model.load_state_dict(ckpt_clean, strict=True)

    # td = torch.load(ckpt_path, map_location="cpu")
    # model.load_state_dict(td)
    # model.relocate()
    # model.eval()
    return model


def load_model(pretrained_weight):
    # load model
    MIL = getattr(models, cfg.Model.base)
    MIL = MIL(**cfg.Model.params)

    new_state_dict = OrderedDict()
    td = torch.load(pretrained_weight, map_location="cpu")
    for key, value in td.items():
        k = key[7:]
        new_state_dict[k] = value

    MIL.load_state_dict(new_state_dict)
    return MIL


def cross_validation(save_csv,pt_dir):
    df = pd.read_csv(save_csv)
    image_ids = []
    probs = []
    labels = []

    for kfold in range(cfg.General.num_folds):
        df_fold = df[df["fold"] == kfold]
        weight_dir = os.path.join(pretrained_weight_root, "fold{}.pth".format(kfold))
        MIL = load_model(weight_dir)

        for image_id in tqdm(df_fold["image_id"].values):
            pt_name = os.path.join(pt_dir, "{}.pt".format(image_id))
            feat = torch.load(pt_name)
            with torch.no_grad():
                logits, Y_prob, Y_hat, results_dict = MIL(feat.cuda())

            Y_prob = Y_prob.squeeze().cpu().numpy()

            image_ids.append(image_id)
            probs.append(Y_prob[1])

    d = {"image_id": image_ids, "prob": probs}
    df = pd.DataFrame(data=d)
    # df.to_csv("./results/{}".format(save_csv), index=None)
    # df = pd.DataFrame(data=d)
    df.to_csv(save_csv, index=None)

def cross_val(save_csv, pt_dir):
    df = pd.read_csv(save_csv)
    num_folds = cfg.General.num_folds

    image_ids = []
    probs_ = []
    probs = []
    labels = []
    hats = []
    for kfold in range(num_folds):
        df_fold = df[df["fold"] == kfold]
        weight_dir = os.path.join(pretrained_weight_root, "fold{}.pth".format(kfold))
        MIL = load_model(weight_dir)
        MIL = MIL.cuda()
        # for image_id in tqdm(df_fold["image_id"].values):
        for row in tqdm(df_fold.values):
            # image_id = row["image_id"]
            # labels.append(row["label"])
            image_id = row[0]
            labels.append(row[1])
            pt_name = os.path.join(pt_dir, "{}.pt".format(image_id))
            # hEP_ft = 
            feat = torch.load(pt_name)
            with torch.no_grad():
                # logits, Y_prob, Y_hat, results_dict = MIL(feat.cuda())
                logits, Y_prob, Y_hat, A_raw, results_dict = MIL(feat.cuda())
            Y_prob = Y_prob.squeeze().cpu().numpy()
            # Y_hat = Y_hat.squeeze().cpu().numpy()

            image_ids.append(image_id)

            probs_.append(Y_prob[1])
            probs.append(int(Y_prob[1].round()))
            hats.append(Y_hat.item())

    auc = roc_auc_score(labels, hats)
    acc = metrics.accuracy_score(labels, hats)
    # f1_b = metrics.f1_score(labels, hats, average='binary') 
    recall = metrics.recall_score(labels, hats) 
    f1 = metrics.f1_score(labels, hats) 

    print(" AUC:",auc)
    print(" acc:",acc)
    print(" F1:",f1)
    print(" recall:",recall)

    d = {"image_id": image_ids, "prob": probs_, "label": labels}
    df = pd.DataFrame(data=d)
    df.to_csv(save_csv, index=None)


def test(save_csv, test_csv, pt_dir):
    df = pd.read_csv(test_csv)
    num_folds = cfg.General.num_folds

    model_list = []
    for fold in range(num_folds):
        # print(fold)
        # weight_dir = os.path.join(pretrained_weight_root, "fold{}.pth".format(fold))

        weight_dir = os.path.join(results_dir, "{}/s_{}_checkpoint.pt".format(exp_code, fold))
        # model.load_state_dict(torch.load(os.path.join(results_dir, "{}/s_{}_checkpoint.pt".format(exp_code, fold)), map_location="cpu"))
        # torch.load(pretrained_weight, map_location="cpu")
        model = load_model_(cfg, weight_dir, fold)
        # model = load_model(weight_dir)
        model_list.append(model)

    image_ids = []
    probs = []
    labels = []
    hats = []
    probs_hats = []


    for idx in tqdm(range(len(df))):
        row = df.loc[idx]
        pt = row.slide_id
        pt_name = os.path.join(pt_dir, "{}.pt".format(pt))
        feat = torch.load(pt_name)
        image_ids.append(pt)
        labels.append(row.label)

        local_y_prob = []
        for MIL in model_list:
            MIL = MIL.cuda()
            with torch.no_grad():
                logits, Y_prob, Y_hat, A_raw, results_dict = MIL(feat.cuda())
        # return logits, Y_prob, Y_hat, A_raw, results_dict

            Y_prob = Y_prob.squeeze().cpu().numpy()
            local_y_prob.append(Y_prob[1])
        # prob = sum(local_y_prob) / len(model_list)
        prob = sum(local_y_prob) / len(local_y_prob)
        probs.append(prob)
        probs_hats.append(prob.round())
        hats.append(Y_hat.item())


    # print('# local_y_prob:', len(local_y_prob))
    print('# probs:', len(probs))
    # print('# hats:', len(hats))
    auc = roc_auc_score(labels, probs)
    acc = metrics.accuracy_score(labels, hats)
    # f1_b = metrics.f1_score(labels, hats, average='binary') 
    recall = metrics.recall_score(labels, hats) 
    f1 = metrics.f1_score(labels, hats) 
    print('# ', test_csv)
    print(" AUC:",auc)
    print(" F1:",f1)
    print(" Acc:",acc)
    print(" recall:",recall)


    d = {"image_id": image_ids, "prob": probs, "label": labels}
    df = pd.DataFrame(data=d)
    df.to_csv(save_csv, index=None)


if __name__=="__main__":
    
    alpha = 0.5
    mu = 0.1
    exp_code  = 'alpha_{}/fl_noise_0.1_mu_{}_50e'.format(alpha, mu)

    data_root_dir =  '/mnt/group-ai-medical-SHARD/private/feifkong/data/prostate_data'
    split_dir  = 'classification_prostate'
    results_dir = '/mnt/group-ai-medical-SHARD/private/feifkong/code_cla/HistoFL_4site_p_v2/results'

    cfg = read_yaml("./configs/config1_test.yaml")
    # val_csv = cfg.General.val_folds
    test1_csv = cfg.General.test1_folds
    test2_csv = cfg.General.test2_folds
    # test3_csv = cfg.General.test3_folds
    # test4_csv = cfg.General.test4_folds
    # test5_csv = cfg.General.test5_folds

    pt_dir1 = cfg.Data.dataset.test_feat_dir
    pt_dir2 = cfg.Data.dataset.test_feat_dir
    # pt_dir3 = cfg.Data.dataset.test_feat_dir
    # pt_dir4 = cfg.Data.dataset.test_feat_dir
    # pt_dir5 = cfg.Data.dataset.test_feat_dir5

    save_csv1= results_dir + '/' + exp_code + "/test_probs_Ds_a.csv"
    save_csv2= results_dir + '/' + exp_code + "/test_probs_qhd.csv"
    # save_csv3= results_dir + '/' + exp_code + "/test_probs_panda.csv"
    # save_csv4= results_dir + '/' + exp_code + "/test_probs_ds_b_hard.csv"
    # save_csv5 = results_dir + '/' + exp_code + "/test_huayin_colorectal.csv"
    print('############# {} ##################'.format(exp_code))

    test(save_csv1, test1_csv, pt_dir1)
    test(save_csv2, test2_csv, pt_dir2)
    # test(save_csv3, test3_csv, pt_dir3)
    # test(save_csv4, test4_csv, pt_dir4)
    # test(save_csv5, test5_csv, pt_dir5)

    