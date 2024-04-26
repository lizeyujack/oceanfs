'''
Author: lizeyujack
Date: 2024-04-15 17:00:14
LastEditors: lizeyujack lizeyujack@163.com
LastEditTime: 2024-04-23 14:15:47
FilePath: /auto-tmp/config.py
Description: 

Copyright (c) 2024 by ${lizeyujack@sjtu.edu.cn}, All Rights Reserved. 
'''
import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Ocean CD-fewshot learning', add_help=False)
    parser.add_argument('--dataset', default='shipsear', type=str, \
                        choices=["deepship_zeroshot_tuning",
                                 "deepship_fewshot_tuning",
                                 "shipsear_zeroshot_tuning",
                                 "shipsear_fewshot_tuning",
                                ],
                        help='dataset name')
    # output_dir default is None
    parser.add_argument('--coop', default='coop', type=str,
                        help='coop type')
    parser.add_argument('--output_dir', default='/cluster/home/lizeyu/oceandil/result', type=str,
                        help='path to save checkpoints and logs')
    # prompt_location default is false,, esle is true string location
    parser.add_argument('--prompt_location', default="None", type=str,
                        help='prompt location')
    parser.add_argument('--ctx_num', default=4, type=int,
                        help='ctx_num number')
    # input_size default is 263670 and not required
    parser.add_argument('--input_size', default=263670, type=int,
                        help='input size of the model')
    #full_model_checkpointing default is None
    parser.add_argument('--full_model_checkpointing', default=None, type=str,
                        help='full model checkpointing')
    # batch_size default is 8
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    # num_workers default is 4
    parser.add_argument('--num_workers', default=16, type=int,  
                        help='number of workers')
    # pin_mem  default is True
    parser.add_argument('--pin_mem', default=True, type=bool,
                        help='pin memory')
    # init_lr default is 0.01
    parser.add_argument('--init_lr', default=0.01, type=float,
                        help='initial learning rate')
    # init_weight_decay default is 0.0001
    parser.add_argument('--init_weight_decay', default=0.0001, type=float,
                        help='initial weight decay')
    # init_epoch default is 100
    parser.add_argument('--init_epoch', default=100, type=int,
                        help='initial epoch')
    # device default is cuda:0
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='device')
    # model_name default is ocean
    parser.add_argument('--output_name', default='ocean_lr_0.001', type=str,
                        help='model name')
    # model_name choose from ['ocean', 'audiomae','ssast']
    parser.add_argument('--model_name', default='ocean', type=str,  choices=['ocean', 'audiomae','ssast','imagebind','pretrain','cross_domain','few_shot','ptuning_lora','ptuning_lora_few_shot','zeroshot','fullzeroshot','clap'],
                        help='model name')
    # random_seed default is 0
    parser.add_argument('--random_seed', default=0, type=int,
                        help='random seed')
    # lora_checkpoint_dir default is '/cluster/home/lizeyu/oceandil/model/ImageBind_LoRA/.checkpoints/lora'
    parser.add_argument('--lora_checkpoint_dir', default='/cluster/home/lizeyu/oceandil/model/ImageBind_LoRA/.checkpoints/lora', type=str,
                        help='lora checkpoint dir')
    # unfreeze_modality default is 1 
    parser.add_argument('--unfreeze_modality', default=1, type=int,
                        help='unfreeze modality')
    # save_epoch default is 10
    parser.add_argument('--save_epoch', default=10, type=int,
                        help='save epoch')
    # resume default is None
    parser.add_argument('--resume', default=None, type=str,
                        help='resume path')
    # n_shot default is 1
    parser.add_argument('--n_shot', default=1, type=int,
                        help='how many shot to use, default is 1')
    # n_way default is 5
    parser.add_argument('--n_way', default=4, type=int,
                        help='how many way to use, default is 5')
    # k_shot default is 5
    parser.add_argument('--k_shot', default=1, type=int,
                        help='how many shot to use, default is 5')
    # q_query default is 15
    parser.add_argument('--q_query', default=5, type=int,
                        help='how many query to use, default is 15')
    # train_episodes default is 10
    parser.add_argument('--train_episodes', default=1, type=int,
                        help='how many train episodes to use, default is 10')
    # test_episodes default is 10
    parser.add_argument('--test_episodes', default=10, type=int,
                        help='how many train episodes to use, default is 10')
    
    return parser


