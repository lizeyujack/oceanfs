
###
 # @Author: lizeyujack
 # @Date: 2024-04-15 10:09:31
 # @LastEditors: lizeyujack lizeyujack@163.com
 # @LastEditTime: 2024-04-15 10:09:34
 # @FilePath: /oceandil/zeroshot.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${lizeyujack@sjtu.edu.cn}, All Rights Reserved. 
### 
python zeroshottuning.py --dataset deepship_zeroshot_tuning \
                        --init_lr 0.001 \
                        --batch_size 2 \
                        --device cuda:0 \
                        --lora_checkpoint_dir /root/autodl-tmp/.checkpoint \
                        --model_name imagebind \
                        --output_dir /cluster/home/lizeyu/oceandil/resulticic24
                        # deepshipの原始 # shipsear_fewshot_tuning #deepship_few_shot_tuning_simple
                        # oceanship
                        # 分类任务。