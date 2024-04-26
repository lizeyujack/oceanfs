'''
Author: lizeyujack
Date: 2024-04-14 21:33:44
LastEditors: lizeyujack lizeyujack@163.com
LastEditTime: 2024-04-23 14:15:30
FilePath: /auto-tmp/model/ocean_coop.py
Description: 

Copyright (c) 2024 by ${lizeyujack@sjtu.edu.cn}, All Rights Reserved. 
'''
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from model.ImageBind.models.multimodal_preprocessors import SimpleTokenizer as _Tokenizer
import numpy as np
from model.ImageBind.data import BPE_PATH
import sys
from utils import TrainerBase
from tqdm import tqdm
import model.ImageBind_LoRA.data as data_
_tokenizer = _Tokenizer(BPE_PATH)


class cfg(object):
    def __init__(self, ctx_num):
        self.backbonename = 'imagebind_huge'
        self.NCTX = ctx_num # 16
        self.CTXINIT = ''
        self.CSC = False
        self.CLASS_TOKEN_POSITION = 'end'
from model.ImageBind.models.helpers import VerboseNNModule

class PromptLearner_imagebind_lora(nn.Module):
    '''
    description: 
    param {*} self
    param {*} cfg
    param {*} classnames
    param {*} ImageBind_model
    return {*}
    '''    
    def __init__(self, cfg, classnames, ImageBind_model):
        super(PromptLearner_imagebind_lora,self).__init__()
        # classnames = ['Fishboat', 'Motorboat', 'Port Tender', 'Spare', 'Trawler', 'Diving ship', 'Dredging', 'Towing', 'Search and Rescue vessel', 'Cargo', 'Pilot ship', 'Tanker', 'Pleasure Craft', 'Passengers', 'RORO', 'Sailboat', 'Military ship', 'Tugboat', 'Ocean liner', 'Mussel boat', 'Law Enforcement', 'Anti-pollution equipment', 'Medical Transport', 'Natural ambient noise', 'Sailing','Dredger']
        self.model = ImageBind_model
        n_cls = len(classnames)
        n_ctx = cfg.NCTX # number of context vectors
        ctx_init = cfg.CTXINIT
        dtype = torch.float32 # torch.float32
        ctx_dim = ImageBind_model.modality_heads.text.proj[1].weight.shape[0]# 1024
        # random initialization
        if cfg.CSC:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype, requires_grad=True)
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, requires_grad=True)# 16, 1024
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.device = ImageBind_model.modality_preprocessors.text.token_embedding.weight.device
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True).to(self.device)
        nn.init.normal_(self.logit_scale, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)#.to(self.device)  # to be optimized
        # nn.init.xavier_uniform_(self.ImageBind_model.text_projection)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "," for name in classnames]# ['X X X X X X X X X X X X X X X X Dredger.', 'X X X X X X X X X X X X X X X X Passengers.', 'X X X X X X X X X X X X X X X X Tugboat.', 'X X X X X X X X X X X X X X X X Ocean liner.', 'X X X X X X X X X X X X X X X X Motorboat.', 'X X X X X X X X X X X X X X X X RORO.', 'X X X X X X X X X X X X X X X X Trawler.', 'X X X X X X X X X X X X X X X X Pilot ship.', 'X X X X X X X X X X X X X X X X Sailboat.', 'X X X X X X X X X X X X X X X X Mussel boat.', 'X X X X X X X X X X X X X X X X Fishboat.', 'X X X X X X X X X X X X X X X X fishboat.', 'X X X X X X X X X X X X X X X X Natural ambient noise.']
        print('prompts',prompts,n_ctx)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION
        
        
        # print('hello',[_tokenizer(p).shape for p in prompts])
        # tokenized_prompts = torch.stack([_tokenizer(p) for p in prompts]).to(self.device)
        # print('self.tokeized_prompty',tokenized_prompts.shape)
        # p_jack = [name for name in classnames]
        # self.tok_p_jack = torch.stack([_tokenizer(p) for p in p_jack]).to(self.device)
        tokenized_prompts = data_.load_and_transform_text(prompts, "cpu")
        # with torch.no_grad():
        #     self.embedding = ImageBind_model.modality_preprocessors['text'](**{"text":self.tok_p_jack})
        self.embedding = ImageBind_model.modality_preprocessors.text.token_embedding(tokenized_prompts.to(self.device))
        # # print('slef. embedding',self.embedding.shape)
        self.register_buffer("token_prefix", self.embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", self.embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        # with torch.no_grad():
        #     embedding = self.model.modality_preprocessors.text(real_tokenized_prompts.to(self.device))
        # attention_mask = self.embedding['trunk']['attn_mask']
        # head = self.embedding['head']
        # embedding = self.embedding['trunk']['tokens']

        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx.to(self.device)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        # return self.embedding_m
        # return {"trunk":{"tokens":self.embedding['trunk']['tokens'],"attn_mask":attention_mask},'head':head}

        text_tokens = prompts
        if self.model.modality_preprocessors.text.num_cls_tokens > 0:
            B = text_tokens.shape[0]
            class_tokens = self.model.modality_preprocessors.text.cls_token.expand(
                B, -1, -1
            )  # stole class_tokens impl from Phil Wang, thanks
            text_tokens = torch.cat((class_tokens, text_tokens), dim=1)
        text_tokens = text_tokens + self.model.modality_preprocessors.text.pos_embed
        return_dict = {
            "trunk": {
                "tokens": text_tokens,
            },
            "head": {},
        }
        # Compute sequence length after adding CLS tokens
        text_lengths = self.tokenized_prompts.argmax(dim=-1)
        return_dict["head"] = {
            "seq_len": text_lengths,
        }
        return return_dict
        # return {"trunk":{"tokens":prompts,"attn_mask":attention_mask},'head':head}

class CustomImagebind(nn.Module):
    def __init__(self, cfg, ImageBind_model,args):
        '''
        description: 少样本处理 imagebind 音频模态和文本模态的scripts 
        return {*}
        '''        
        super().__init__()
        if args.dataset=='shipsear_fewshot_tuning':classnames = ['Fishboat', 'Motorboat', 'Port Tender', 'Spare', 'Trawler', 'Diving ship', 'Dredging', 'Towing', 'Search and Rescue vessel', 'Cargo', 'Pilot ship', 'Tanker', 'Pleasure Craft', 'Passengers', 'RORO', 'Sailboat', 'Military ship', 'Tugboat', 'Ocean liner', 'Mussel boat', 'Law Enforcement', 'Anti-pollution equipment', 'Medical Transport', 'Natural ambient noise', 'Sailing','Dredger']
        elif args.dataset=='deepship_fewshot_tuning':classnames = ['Cargo', 'Tanker', 'Passenger', 'Tug']
        p_jack = [name for name in classnames]
        
        self.prompt_learner = PromptLearner_imagebind_lora(cfg, classnames, ImageBind_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ImageBind_model = ImageBind_model
        self.dtype = ImageBind_model.modality_preprocessors.text.token_embedding.weight.dtype
        self.device = ImageBind_model.modality_preprocessors.text.token_embedding.weight.device
        self.tok_p_jack = torch.stack([_tokenizer(p) for p in p_jack]).to(self.device)
    def audio_encoder(self, audio):
        if audio.ndim == 5:
            B, S = audio.shape[:2]
            audio = audio.reshape(
                B * S, *audio.shape[2:]
            )
        modality_value = self.ImageBind_model.modality_preprocessors['audio'](audio=audio)
        trunk_inputs = modality_value["trunk"]
        head_inputs = modality_value["head"]
        modality_value = self.ImageBind_model.modality_trunks['audio'](**trunk_inputs)
        modality_value = self.ImageBind_model.modality_heads['audio'](
            modality_value, **head_inputs
        )
        modality_value = self.ImageBind_model.modality_postprocessors['audio'](
            modality_value
        )
        modality_value = modality_value.reshape(B, S, -1)
        modality_value = modality_value.mean(dim=1)
        return modality_value

    def text_encoder(self, embedding):# nan label here
        trunk_inputs = embedding["trunk"]
        head_inputs = embedding["head"]

        modality_value = self.ImageBind_model.modality_trunks['text'](**trunk_inputs)
        modality_value = self.ImageBind_model.modality_heads['text'](
            modality_value, **head_inputs
        )
        modality_value = self.ImageBind_model.modality_postprocessors['text'](
            modality_value
        )
        return modality_value

    def audio_embedding(self, audio):
        return self.audio_encoder(audio)

    def text_embedding(self):
        embedding = self.prompt_learner()
        text_features = self.text_encoder(embedding)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

def freezeall(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def nameparameter(model):
    for name, param in model.named_parameters():
        print(name)
from model.CoOp.Dassl.dassl.utils import Registry, check_availability
TRAINER_REGISTRY = Registry("TRAINER")
def analysis_parameter(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params}")
    # print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    print('loaded model lora')
    print('8'*8)

@TRAINER_REGISTRY.register()
class CoOp(TrainerBase):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    # def check_cfg(self, cfg):
    #     assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def __init__(self, args):
        super().__init__() 
        from oceannet_pretrain_lora import imagebind_lora
        from utils import remove_useless_keys
        import model.ImageBind.models.imagebind_model as imagebind_model
        if args.model_name == 'imagebind':# version 1 imagebind版本
            model = imagebind_model.imagebind_huge(pretrained=True)
            # load tuned model
            model = remove_useless_keys(model)

            if args.lora_checkpoint_dir:
                print('loading from lora_checkpoint_dir')
                model = imagebind_lora(model, lora_checkpoint_dir=args.lora_checkpoint_dir)
            model = freezeall(model)
            # nameparameter(model)
            analysis_parameter(model)
            model = model.to(args.device)
            self.model = CustomImagebind(cfg(args.ctx_num),model,args).to(args.device)
            self.orimodel = self.model.ImageBind_model
            analysis_parameter(self.model)


        if args.model_name == 'clap':
            from model.CLAP.src import laion_clap
            self.model = laion_clap.CLAP_Module(enable_fusion=False)
            self.model.load_ckpt() # download the default pretrained checkpoint.
        for name, param in self.model.named_parameters():
            if name == 'prompt_learner.ctx':
                param.requires_grad_(True)
        from model.CoOp.Dassl.dassl.utils.torchtools import load_pretrained_weights,load_checkpoint
        from model.CoOp.Dassl.dassl.optim.lr_scheduler import ConstantWarmupScheduler
        if args.prompt_location != 'None':
            load_pretrained_weights(model.prompt_learner, args.prompt_location)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print('param', name)
        # sys.exit()
        # self.optim = torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=args.init_lr,
        #     weight_decay=1e-2,
        #     betas=(0.9, 0.999)
        # )
        self.optim = torch.optim.SGD(
            self.model.parameters(),
            lr=args.init_lr,
            momentum = 0.9,
            weight_decay=5e-2,
            dampening=0,
            nesterov=False,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, float(100)
        )
        self.sched = ConstantWarmupScheduler(
                self.optim, scheduler, 20,# epochs of warm up step
                1e-5# learning rate
            )
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        # self.scaler = GradScaler()
    def run_one_epoch(self, data, metrics, args, mode):
        length = len(data)
        if mode == 'train':
            for batch_idx, (audio, img, text, label) in tqdm(enumerate(data),total=length):
                loss, metrics = self.train_backward(audio, text, label, batch_idx, metrics, args)
            return loss, metrics
        elif mode == 'test':
            for batch_idx, (audio, img, text, label) in tqdm(enumerate(data),total=length):
                metrics = self.test_backward(audio, text, label, batch_idx, metrics, args)
            return metrics
        else:
            raise ValueError("Invalid value")
    def test_backward(self, audio, text, label, batch_idx, metrics, args):
        from model.CoOp.Dassl.dassl.metrics.accuracy import compute_accuracy
        if args.model_name == 'imagebind':
            audio = audio.type(self.model.dtype)
            audio = audio.to(args.device)
            text = text[0].squeeze(0).to(args.device)
            self.model = self.model.to(args.device)
            feats_a_tensor = self.orimodel({"audio": audio})['audio']
            feats_b_tensor = self.model.text_embedding()
        if args.model_name == 'clap':# 加载内容应为纯文本或者音频。不可以为预处理好的tensor格式(dict_.values())[0] for dict_ in feats_b], dim=0)# 文本text
            feats_a_tensor = self.model.getaudioembedding(audio)
            feats_b_tensor = self.model.gettextembedding(text)
        label = label.to(args.device)
        logits_audio_text = feats_a_tensor@ feats_b_tensor.T
        try:
            predicted = torch.argmax(logits_audio_text.squeeze(1), dim=1)
        except:
            predicted = torch.argmax(logits_audio_text, dim=1)
        for metric in metrics:
            metrics[metric].update(predicted, label.to(predicted.device))
        return metrics

    def train_backward(self, audio, text, label, batch_idx, metrics, args):
        from model.CoOp.Dassl.dassl.metrics.accuracy import compute_accuracy
        if args.model_name == 'imagebind':
            audio = audio.type(self.model.dtype)
            audio = audio.to(args.device)
            text = text[0].squeeze(0).to(args.device)
            self.model = self.model.to(args.device)
            feats_a_tensor = self.orimodel({"audio": audio})['audio']
            feats_b_tensor = self.model.text_embedding()

        if args.model_name == 'clap':# 加载内容应为纯文本或者音频。不可以为预处理好的tensor格式(dict_.values())[0] for dict_ in feats_b], dim=0)# 文本text
            feats_a_tensor = self.model.getaudioembedding(audio)
            feats_b_tensor = self.model.gettextembedding(text)
        label = label.to(args.device)
        logits_audio_text = self.model.prompt_learner.logit_scale*feats_a_tensor@ feats_b_tensor.T
        try:
            predicted = torch.argmax(logits_audio_text.squeeze(1), dim=1)
        except:
            predicted = torch.argmax(logits_audio_text, dim=1)
        loss = F.cross_entropy(predicted.float(), label.float())
        names = self.get_model_names()
        self.model_backward_and_update(loss,names)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits_audio_text, label)[0].item(),
        }
        for metric in metrics:
            metrics[metric].update(predicted, label.to(predicted.device))
        # print('acc',loss_summary['acc'])
        self.update_lr()
        return loss_summary['loss'], metrics

    def load_model(self, directory, epoch=None):
        from model.CoOp.Dassl.dassl.utils.torchtools import load_checkpoint

        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"# 导入模型

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors #
            if "token_prefix" in state_dict:#prompt中的参数
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)