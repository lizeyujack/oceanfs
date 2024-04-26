from model.ImageBind.data import load_and_transform_text, load_and_transform_vision_data, load_and_transform_audio_data
import glob, os
from tqdm import tqdm
import torch
list_fk = glob.glob("/cluster/home/lizeyu/oceandil/dataset/shipsear/cut_5s/*/*/*.wav")
for _, audio in tqdm(enumerate(list_fk), total=len(list_fk)):
    feather = load_and_transform_audio_data(audio,"cuda:2")
    fielname = os.path.basename(audio)
    filedir = os.path.dirname(audio)
    torch.save(feather, os.path.join(filedir, fielname[:-4]+".pt"))
    # print('feather',feather.shape)