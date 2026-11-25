from g2_force_act_model import make_model as base_model,normalize_images
def make_model(*args,**kwargs):
 model=base_model(*args,**kwargs)
 # Corrective demonstrations provide real off-policy observations. Train on
 # those observations without the previous observation-masking ablation.
 assert len(model._forward_pre_hooks)==1
 assert next(iter(model._forward_pre_hooks.values())).__name__=='mask_observations'
 model._forward_pre_hooks.clear()
 return model
