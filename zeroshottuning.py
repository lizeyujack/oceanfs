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
from config import get_args_parser
from trainer import ocean_Trainer, Imagebind_inferencer, load_pretrained_model
from utils import remove_useless_keys
from oceannet_few_shot import oceannet_few_shot_inference
from oceannet_zero_shot import oceannet_zero_shot_inference
# from oceannet_train import oceannet_train
from utils import load_dataset
from oceannet_pretrain_lora import imagebind_lora
from model.oceannet import Ocean_net, oceannet_lora
import sys
from tqdm import tqdm
from trainer import ocean_Trainer, inference, metric_tool
def main(args): 
    print("dataset",args.dataset)
    build_dataset, classname = load_dataset(args)
    train_loader, val_loader,nb_classes = build_dataset(args)
    print('starting to load')
    # oceannet_few_shot_inference(val_loader, train_loader, nb_classes, ocean_Trainer, Ocean_net, imagebind_model, remove_useless_keys, classname, args)
    # model = imagebind_model.imagebind_huge(pretrained=True)# 参考预训练部分的代码。
    if args.model_name == 'imagebind':
        model = imagebind_model.imagebind_huge(pretrained=True)
        # print(model)
        # sys.exit()
        # load tuned model
        # model.load_state_dict(torch.load("/cluster/home/lizeyu/oceandil/.checkpoints/cdoceanship/last.ckpt", map_location=torch.device('cpu')), strict=False)
        model = remove_useless_keys(model)
        if args.lora_checkpoint_dir:model = imagebind_lora(model, lora_checkpoint_dir=args.lora_checkpoint_dir)
        print('loaded model lora')
        print('args.device',args.device)
    if args.model_name == 'clap':
        from model.CLAP.src import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt() # download the default pretrained checkpoint.
        # audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=False)
        # break
        

    model.to(args.device)
    metrics = metric_tool(nb_classes=nb_classes, args=args)
    tqdm_batch = tqdm(total=len(val_loader), desc= 'Test_Batch', leave= False)
    acc_list, predict_list, label_list = [], [], []
    audio_all, label_all = list(), list()
    for batch_idx, (text, audio, img, label) in enumerate(val_loader):
        if args.model_name == 'imagebind':
            # audio = audio.unsqueeze(1).to(model.modality_preprocessors.audio.cls_token.device)
            print('audio',audio.shape)
            print('text',text.shape)
            # sys.exit()
            text = text.squeeze(1).to(model.modality_preprocessors.audio.cls_token.device)
            text = text[0].squeeze(0)
            audio = audio.squeeze(1).to(model.modality_preprocessors.audio.cls_token.device)
            print('audio pro',audio.shape)
            sys.exit()
            # feats_a = [model({"audio": data_a_i}) for data_a_i in audio]
            audio_embedding = model({"audio": audio})
            # # print('audio_embedding',audio_embedding['audio'].shape)
            text_embedding = model({"text": text})
            
            # feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0) 
            # print('audio_embedding',audio_embedding['audio'].shape)
            feats_a_tensor = audio_embedding['audio']
            # print('text_embedding',text_embedding['text'].shape)
            feats_b_tensor = text_embedding['text']
            # sys.exit()
            # audio_all.append(feats_a_tensor.detach().cpu())
            # label_all.append(label.detach().cpu())
            # feats_b = [model({"text": data_b_i}) for idx, data_b_i in enumerate(text[0])]
            # feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)# 文本text
        if args.model_name == 'clap':# 加载内容应为纯文本或者音频。不可以为预处理好的tensor格式(dict_.values())[0] for dict_ in feats_b], dim=0)# 文本text
            feats_a_tensor = model.getaudioembedding(audio,img)
            feats_b_tensor = model.gettextembedding(text)
        audio_all.append(feats_a_tensor.detach().cpu())
        label_all.append(label.detach().cpu())
        logits_audio_text = feats_a_tensor@ feats_b_tensor.T
        print('logits_audio_text',logits_audio_text.shape)
        print('label',label.shape)
        try:
            predicted = torch.argmax(logits_audio_text.squeeze(1), dim=1)
        except:
            predicted = torch.argmax(logits_audio_text, dim=1)
        tqdm_batch.update()
        for metric in metrics:
            metrics[metric].update(predicted, label.to(predicted.device))
    audio_features = torch.cat(audio_all)
    labels = torch.cat(label_all)
    
    feats_b_tensor = feats_b_tensor.detach().cpu()
    logits_audio_text = audio_features@ feats_b_tensor.T

    # text_features = torch.cat(label_all)
    ranking = torch.argsort(logits_audio_text, descending=True)
    print('ranking\t',ranking, ranking.shape)
    print('lable\t',labels, labels.shape)
    preds = torch.nonzero(torch.eq(ranking, labels.unsqueeze(1).to(ranking.device)))[:, 1]
    preds = preds.cpu().numpy()
    # # print('preds',preds.shape,preds)
    rmetrics = {}
    rmetrics[f"mean_rank"] = preds.mean() + 1
    rmetrics[f"median_rank"] = np.floor(np.median(preds)) + 1
    for k in [1,3, 5, 10]:
        rmetrics[f"R@{k}"] = np.mean(preds < k)
    
    metrics_dic = {}
    for idx, metric in enumerate(metrics):
        value = metrics[metric].compute()
        # print(metric, value)
        metrics_dic[metric] = value
    
    for idx, metric in enumerate(rmetrics):
        metrics_dic[metric] = rmetrics[metric]
        print(metric,rmetrics[metric])
    acc = metrics_dic["R@1"]
    tqdm_batch.reset()
    
    
    # print('type(args)',type(args))
    # 将acc_list和total写入txt文件内
    with open(os.path.join(args.output_dir, f'{args.dataset}_{os.path.basename(args.lora_checkpoint_dir)}_acc_{float(acc)}.txt'), 'w') as f:
        for value in metrics_dic:
            f.write(value+":\t"+str(metrics_dic[value])+ "\n")
    f.close()
        # return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Ocean CD-fewshot learning', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    