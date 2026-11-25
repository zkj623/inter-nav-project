import torch
from g2_act_model import make_model as base_model, normalize_images
def make_model(chunk=50,hidden=512,state_skip=False,time_conditioned=True):
 assert time_conditioned and not state_skip
 model=base_model(chunk,hidden,time_conditioned=True)
 model.progress_encoder=torch.nn.Sequential(torch.nn.Linear(17,128),torch.nn.SiLU(),torch.nn.Linear(128,hidden),torch.nn.LayerNorm(hidden))
 def progress(module,inputs,output):
  t=inputs[0][:,-1:]
  freq=2.**torch.arange(8,device=t.device,dtype=t.dtype)*torch.pi/8
  features=torch.cat([t/8,torch.sin(t*freq),torch.cos(t*freq)],dim=-1)
  return output+model.progress_encoder(features)
 model.input_proj_robot_state.register_forward_hook(progress)
 def mask_observations(module,inputs):
  if not module.training:return inputs
  q,image,*other=inputs
  q=q.clone();image=image.clone()
  q[:,:16]*=(torch.rand((len(q),1),device=q.device)>.5)
  image*=(torch.rand((len(q),1,1,1,1),device=image.device)>.3)
  return (q,image,*other)
 model.register_forward_pre_hook(mask_observations)
 return model
