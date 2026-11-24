from pathlib import Path
from types import SimpleNamespace
import sys,torch
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'vendor/act/detr'))
from models.detr_vae import build
def make_model(chunk=50,hidden=512):
 args=SimpleNamespace(hidden_dim=hidden,dropout=0.,nheads=8,dim_feedforward=3200,enc_layers=4,dec_layers=7,pre_norm=False,num_queries=chunk,camera_names=['head_front','wrist_left','wrist_right'],backbone='resnet18',lr_backbone=1e-5,masks=False,dilation=False,position_embedding='sine')
 return build(args)
def normalize_images(images):
 mean=images.new_tensor([.485,.456,.406]).view(1,1,3,1,1);std=images.new_tensor([.229,.224,.225]).view(1,1,3,1,1)
 return (images-mean)/std
