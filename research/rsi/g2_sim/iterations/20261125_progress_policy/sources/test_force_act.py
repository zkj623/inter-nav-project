from pathlib import Path
import json,os
import numpy as np,torch
from g2_force_features import causal_force,encode_force,FINGER_NAMES
from g2_force_act_model import make_model
from g2_progress_act_model import make_model as previous_model
W=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.manual_seed(17)
samples=[dict(physics_time_s=t,finger_forces=dict.fromkeys(FINGER_NAMES,v)) for t,v in [(1.,1.),(1.1,2.),(1.2,6.),(1.3,100.)]]
values,stamp=causal_force(samples,1.25)
assert np.allclose(values,[3.]*4) and stamp==1.2
assert np.array_equal(values,causal_force(samples[:-1],1.25)[0])
for now in [.9,1.5]:
 try:causal_force(samples,now)
 except ValueError:pass
 else:raise AssertionError('Missing/stale samples must fail')
assert np.isfinite(encode_force([0,1,35,1000])).all()
state=torch.load(W/'runs/act_progress_v1/step_008000.pt',map_location='cpu',weights_only=False)['model']
old=previous_model();old.load_state_dict(state);old.eval()
new=make_model();expanded={k:v.clone() for k,v in state.items()}
for key in ['input_proj_robot_state.weight','encoder_joint_proj.weight']:
 expanded[key]=torch.zeros_like(new.state_dict()[key]);expanded[key][:,:16]=state[key][:,:16];expanded[key][:,-1]=state[key][:,-1]
missing=new.load_state_dict(expanded,strict=False)
assert not missing.unexpected_keys and all(k.startswith('force_encoder.') for k in missing.missing_keys)
new.eval();q=torch.randn(2,17);force=torch.rand(2,4);q21=torch.cat([q[:,:16],force,q[:,-1:]],dim=1);images=torch.randn(2,3,3,64,64);tasks=torch.tensor([3,4])
with torch.no_grad():
 reference=old(q,images,None,task_ids=tasks)[0];actual=new(q21,images,None,task_ids=tasks)[0]
assert torch.allclose(reference,actual,atol=1e-5),float((reference-actual).abs().max())
del old,state,expanded
new.zero_grad();out=new(q21,images,None,task_ids=tasks)[0];out.square().mean().backward()
assert new.force_encoder[-1].weight.grad.abs().sum()>0
with torch.no_grad():
 new.force_encoder[-1].weight.add_(new.force_encoder[-1].weight.grad,alpha=-.1)
 changed=q21.clone();changed[:,16:20]+=1.
 effect=float((new(q21,images,None,task_ids=tasks)[0]-new(changed,images,None,task_ids=tasks)[0]).abs().max())
assert effect>1e-7
result=dict(past_only_alignment=True,stale_observation_rejected=True,source_checkpoint_predictions_preserved=True,finite_output=bool(torch.isfinite(out).all()),shape=list(out.shape),contact_gradient_nonzero=True,contact_input_effect_after_update=effect)
(W/'force_model_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
