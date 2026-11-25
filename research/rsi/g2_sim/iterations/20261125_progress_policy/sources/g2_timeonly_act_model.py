import torch
from g2_progress_act_model import make_model as progress_model,normalize_images
def make_model(*args,**kwargs):
 model=progress_model(*args,**kwargs)
 def mask(module,inputs):
  q,image,*other=inputs;q=q.clone();q[:,:16]=0
  return (q,torch.zeros_like(image),*other)
 model.register_forward_pre_hook(mask)
 return model
