import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from model.ImageBind.models.multimodal_preprocessors import SimpleTokenizer as _Tokenizer
import numpy as np
from model.ImageBind.data import BPE_PATH
import sys
from utils import TrainerBase
from model.utils import freezeall, analysis_parameter
from tqdm import tqdm
import model.ImageBind_LoRA.data as data_
_tokenizer = _Tokenizer(BPE_PATH)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LogitHead(nn.Module):
    def __init__(self, head, logit_scale=float(np.log(1 / 0.07))):
        super().__init__()
        self.head = head
        self.logit_scale = logit_scale
        
        # Not learnable for simplicity
        self.logit_scale = torch.FloatTensor([logit_scale]).cuda()
        # Learnable
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * logit_scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.head(x)
        x = x * self.logit_scale.exp()
        return x

class MM_linear:
    def __init__(self,args):
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
        from model.CoOp.Dassl.dassl.optim.lr_scheduler import ConstantWarmupScheduler
        self.text_label = torch.tensor([i for i in range(4)])
        logit_head = LogitHead(deepcopy(head), logit_scale=logit,)
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
    def train_backward(self,audio, text, label, batch_idx, metrics, args):
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
        feature = torch.cat([feats_a_tensor, feats_b_tensor], dim=0)
        label = torch.cat([label, self.text_label], dim=0)
        logit = logit_head(feature)
        loss = criterion(logit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return 0
        # logits_audio_text = feats_a_tensor@ feats_b_tensor.T
        # try:
        #     predicted = torch.argmax(logits_audio_text.squeeze(1), dim=1)
        # except:
        #     predicted = torch.argmax(logits_audio_text, dim=1)
        # loss = F.cross_entropy(predicted.float(), label.float())
        # names = self.get_model_names()
        # self.model_backward_and_update(loss,names)

        # loss_summary = {
        #     "loss": loss.item(),
        #     "acc": compute_accuracy(logits_audio_text, label)[0].item(),
        # }
        # for metric in metrics:
        #     metrics[metric].update(predicted, label.to(predicted.device))
        # # print('acc',loss_summary['acc'])
        # self.update_lr()
        # return loss_summary['loss'], metrics

    def train(logit_head, image_encoder, text_encoder,
            image_loader, val_loader, text_loader,
            optimizer, scheduler, criterion, iters,
            eval_freq=EVAL_FREQ, device="cuda"):
        if image_loader is None and text_loader is None:
            raise ValueError("Both image_loader and text_loader are None")
        if image_loader is not None:
            image_loader_iter = iter(image_loader)
        else:
            image_loader_iter = None
        if text_loader is not None:
            text_loader_iter = iter(text_loader)
        else:
            text_loader_iter = None

        result_dict = {
            "iter": None,
            "val_acc": None,
            "image_encoder": None,
            "text_encoder": None,
            "logit_head": None,
        }

        for i in range(iters):
            logit_head.train()
            image_encoder.train()
            text_encoder.train()
            if image_loader_iter is not None:
                try:
                    image, image_label = next(image_loader_iter)
                except StopIteration:
                    image_loader_iter = iter(image_loader)
                    image, image_label = next(image_loader_iter)
                image = image.to(device)
                image_label = image_label.to(device)
                image_feature = image_encoder(image)
            else:
                image_feature = None
            
            if text_loader_iter is not None:
                try:
                    text, text_label, eot_indices = next(text_loader_iter)
                except StopIteration:
                    text_loader_iter = iter(text_loader)
                    text, text_label, eot_indices = next(text_loader_iter)
                text = text.to(device)
                text_label = text_label.to(device)
                eot_indices = eot_indices.to(device)
                text_feature = text_encoder(text, eot_indices)
            else:
                text_feature = None
            
            if image_feature is not None and text_feature is not None:
                feature = torch.cat([image_feature, text_feature], dim=0)
                label = torch.cat([image_label, text_label], dim=0)
            elif image_feature is not None:
                feature = image_feature
                label = image_label
            elif text_feature is not None:
                feature = text_feature
                label = text_label
            else:
                raise ValueError("Both image_feature and text_feature are None")

            logit = logit_head(feature)
            loss = criterion(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if i % eval_freq == 0:
                val_acc = validate(logit_head, image_encoder, val_loader, device=device)
                if result_dict["val_acc"] is None or val_acc > result_dict["val_acc"]:
                    result_dict["iter"] = i
                    result_dict["val_acc"] = val_acc
                    result_dict["image_encoder"] = deepcopy(image_encoder.state_dict())
                    result_dict["text_encoder"] = deepcopy(text_encoder.state_dict())
                    result_dict["logit_head"] = deepcopy(logit_head.state_dict())
        
        # load best model
        image_encoder.load_state_dict(result_dict["image_encoder"])
        text_encoder.load_state_dict(result_dict["text_encoder"])
        logit_head.load_state_dict(result_dict["logit_head"])
        val_acc = validate(logit_head, image_encoder, val_loader, device=device)
        print(f"Best val acc: {result_dict['val_acc']:.4f} at iter {result_dict['iter']}")
        return result_dict