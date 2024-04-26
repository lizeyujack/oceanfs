# -*- encoding: utf-8 -*-
'''
@File : main_inference.py
@Date : 2023-06-17
@Time : 19:25:18
@Author : lizeyujack@sjtu.edu.cn 
'''
import argparse,sys
import os
import numpy as np
from pathlib import Path
import model.ImageBind.models.imagebind_model as imagebind_model
import torch
from model.oceannet import Ocean_net
from model.ocean_coop import CoOp
from model.ocean_cocoop import CoCoOp
from config import get_args_parser
from trainer import ocean_Trainer, Imagebind_inferencer, load_pretrained_model
from utils import remove_useless_keys
from oceannet_few_shot import oceannet_few_shot_inference
from oceannet_zero_shot import oceannet_zero_shot_inference
from torch.nn import functional as F
# from oceannet_train import oceannet_train
from utils import load_dataset
from oceannet_pretrain_lora import imagebind_lora
from model.oceannet import Ocean_net, oceannet_lora
import sys
from tqdm import tqdm
from trainer import ocean_Trainer, inference, metric_tool
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import time
import datetime

def metric_tool(nb_classes, args):
    num_of_classes = nb_classes
    if type(args) == argparse.Namespace:args = vars(args)
    device = args['device']
    accuracy = Accuracy(average='macro', num_classes=num_of_classes).to(device)
    accuracy_micro = Accuracy(average='micro', num_classes=num_of_classes).to(device)
    accuracy_weight = Accuracy(average='weighted', num_classes=num_of_classes).to(device)
    precision = Precision(average='macro', num_classes=num_of_classes).to(device)
    recall = Recall(average='macro', num_classes=num_of_classes).to(device)
    f1 = F1Score(average='macro', num_classes=num_of_classes).to(device)
    confusion_matrix = ConfusionMatrix(num_classes=num_of_classes).to(device)
    metrics = {
        "Accuracy":accuracy,
        "AccuracyMicro":accuracy_micro,
        "AccuracyWeighted":accuracy_weight,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "ConfusionMatrix":confusion_matrix,
    }
    return metrics

def show_result(mets,args, tuning_set):
    timestamp = time.time()
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    time_formatted = dt_object.strftime('%y-%m-%d %H:%M:%S')
    metrics_dic = {}
    for idx, metric in enumerate(mets):
        value = mets[metric].compute()
        metrics_dic[metric] = value
    print(metrics_dic)
    save_path = os.path.join(args.output_dir,'result.txt')
    if tuning_set != 'eval':
        with open(save_path, "a") as f:
            placeholder = '=' * 8
            f.write(f'{placeholder}{args.dataset} in {args.model_name} with {args.n_shot}-shot in {time_formatted} 2024{placeholder}\n')
            f.write(tuning_set+"\n")
            for keyi in metrics_dic:
                f.write(f"{keyi} {metrics_dic[keyi]}\n")
        f.close()
    else:
        pass
    return metrics_dic


def main(args): 
    print("dataset",args.dataset)
    build_dataset, classname = load_dataset(args)
    train_loader, val_loader,nb_classes = build_dataset(args)
    
    print('starting to load')
    if args.model_name == 'imagebind':# version 1 imagebind版本
        if args.coop == 'coop':
            print('loading coop')
            coop = CoOp(args)# v2
        elif args.coop == 'cocoop':
            print('loading cocoop')
            coop = CoCoOp(args)
    if args.model_name == 'clap':
        # from model.CLAP.src import laion_clap
        # model = laion_clap.CLAP_Module(enable_fusion=False)
        # model.load_ckpt() # download the default pretrained checkpoint.
        pass
    # training step
    metrics = metric_tool(nb_classes=nb_classes, args=args)
    mets = coop.run_one_epoch(val_loader, metrics, args,'test')
    show_result(mets,args,'before tuning')
    for idx in range(0, args.init_epoch):
        # refresh everytime u run each epoch
        metrics = metric_tool(nb_classes=nb_classes, args=args)
        loss, metrics = coop.run_one_epoch(train_loader, metrics, args,'train')
        print('loss',loss)
        if idx % 1 == 0:
            metrics_eval = metric_tool(nb_classes=nb_classes, args=args)
            mets = coop.run_one_epoch(val_loader, metrics_eval, args, 'test')
            show_result(mets,args,'after tuning')
    metrics = metric_tool(nb_classes=nb_classes, args=args)
    mets = coop.run_one_epoch(val_loader, metrics, args,'test')
    show_result(mets,args,'after tuning')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Ocean CD-fewshot learning', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    