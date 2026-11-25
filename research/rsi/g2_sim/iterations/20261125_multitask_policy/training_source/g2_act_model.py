from pathlib import Path
from types import SimpleNamespace
import sys,torch
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'vendor/act/detr'))
from models.detr_vae import build
def make_model(chunk=50,hidden=512,state_skip=False,time_conditioned=False):
 args=SimpleNamespace(hidden_dim=hidden,dropout=0.,nheads=8,dim_feedforward=3200,enc_layers=4,dec_layers=7,pre_norm=False,num_queries=chunk,camera_names=['head_front','wrist_left','wrist_right'],backbone='resnet18',lr_backbone=1e-5,masks=False,dilation=False,position_embedding='sine')
 model=build(args)
 if time_conditioned:
  assert not state_skip,'Time baseline and state-skip ablation are separate models'
  for name in ['input_proj_robot_state','encoder_joint_proj']:
   old=getattr(model,name);new=torch.nn.Linear(17,hidden)
   with torch.no_grad():new.weight[:,:16].copy_(old.weight);new.bias.copy_(old.bias)
   setattr(model,name,new)
 if state_skip:
  model.state_action_skip=torch.nn.Linear(16,16,bias=False)
  torch.nn.init.zeros_(model.state_action_skip.weight)
  def add_state(module,inputs,output):return (output[0]+module.state_action_skip(inputs[0])[:,None,:],output[1],output[2])
  model.register_forward_hook(add_state)
 return model
def normalize_images(images):
 mean=images.new_tensor([.485,.456,.406]).view(1,1,3,1,1);std=images.new_tensor([.229,.224,.225]).view(1,1,3,1,1)
 return (images-mean)/std


