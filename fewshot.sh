
###
 # @Author: lizeyujack
 # @Date: 2024-04-17 18:24:39
 # @LastEditors: lizeyujack lizeyujack@163.com
 # @LastEditTime: 2024-04-23 14:46:38
 # @FilePath: /auto-tmp/fewshot.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${lizeyujack@sjtu.edu.cn}, All Rights Reserved. 
### 
for SHOT in 1 3 5 7 9 10
do
    python fewshottuning.py --dataset deepship_fewshot_tuning \
                            --n_shot ${SHOT} \
                            --init_lr 2e-3 \
                            --batch_size 8 \
                            --init_epoch 50 \
                            --device cuda:0 \
                            --lora_checkpoint_dir /root/autodl-tmp/.checkpoint/ \
                            --model_name imagebind \
                            --output_dir /root/autodl-tmp/resulticic24/ \
                            --ctx_num 2 \
                            --coop cocoop >> cocoop_${SHOT}shot_ctx3.txt
done                        

for SHOT in 1 3 5 7 9 10
do
    python fewshottuning.py --dataset deepship_fewshot_tuning \
                            --n_shot ${SHOT} \
                            --init_lr 2e-3 \
                            --batch_size 8 \
                            --init_epoch 50 \
                            --device cuda:0 \
                            --lora_checkpoint_dir /root/autodl-tmp/.checkpoint/ \
                            --model_name imagebind \
                            --output_dir /root/autodl-tmp/resulticic24/ \
                            --ctx_num 2 \
                            --coop coop >> coop_${SHOT}shot_ctx3.txt
done                        