import torch
from g2_progress_act_model import make_model as base_model,normalize_images
def make_model(chunk=50,hidden=512,state_skip=False,time_conditioned=True):
 model=base_model(chunk,hidden,state_skip,time_conditioned)
 for name in ['input_proj_robot_state','encoder_joint_proj']:
  old=getattr(model,name);new=torch.nn.Linear(21,hidden)
  with torch.no_grad():
   new.weight.zero_();new.weight[:,:16].copy_(old.weight[:,:16]);new.weight[:,-1].copy_(old.weight[:,-1]);new.bias.copy_(old.bias)
  setattr(model,name,new)
 # Replacing the state projection removes its original Fourier hook; restore it.
 def progress(module,inputs,output):
  q=inputs[0];t=q[:,-1:];freq=2.**torch.arange(8,device=t.device,dtype=t.dtype)*torch.pi/8
  features=torch.cat([t/8,torch.sin(t*freq),torch.cos(t*freq)],dim=-1)
  return output+model.progress_encoder(features)+model.force_encoder(q[:,16:20])
 model.force_encoder=torch.nn.Sequential(torch.nn.Linear(4,64),torch.nn.SiLU(),torch.nn.Linear(64,hidden))
 torch.nn.init.zeros_(model.force_encoder[-1].weight);torch.nn.init.zeros_(model.force_encoder[-1].bias)
 model.input_proj_robot_state.register_forward_hook(progress)
 return model
