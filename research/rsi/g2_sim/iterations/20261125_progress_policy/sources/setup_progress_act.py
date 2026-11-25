from pathlib import Path
import hashlib,json,py_compile
W=Path(__file__).resolve().parent
model='''import torch
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
'''
(W/'g2_progress_act_model.py').write_text(model)
src=(W/'train_g2_act.py').read_text(encoding='utf-8-sig').replace('from g2_act_model import','from g2_progress_act_model import')
old="set(incompatible.missing_keys)<=({'state_action_skip.weight'} if a.state_skip else set())"
assert old in src
src=src.replace(old,"all(k.startswith('progress_encoder.') for k in incompatible.missing_keys)")
old="if band<.15:t=np.random.randint(min(20,n))\n   elif band<.5 and len(contact):t=int(np.random.choice(contact))"
assert old in src
src=src.replace(old,"if band<.10:t=np.random.randint(min(20,n))\n   elif band<.30 and len(contact):t=int(np.random.choice(contact))\n   elif band<.65:\n    phases=f['phase'].asstr()[:];boundaries=np.flatnonzero(phases[1:]!=phases[:-1])+1\n    if len(boundaries):t=int(np.clip(np.random.choice(boundaries)+np.random.randint(-25,51),0,n-1))")
src=src.replace("W/'g2_act_model.py',","W/'g2_act_model.py',W/'g2_progress_act_model.py',")
(W/'train_progress_act.py').write_text(src)
server=(W/'act_policy_server.py').read_text(encoding='utf-8-sig').replace('from g2_act_model import','from g2_progress_act_model import').replace("'ACT_elapsed_time'","'ACT_progress_conditioned'")
(W/'progress_act_policy_server.py').write_text(server)
guard=(W/'resume_guarded_eval.py').read_text()
guard=guard.replace("OUT=W/'recovery_20261125'","OUT=W/'progress_act_20261125'")
guard=guard.replace("lock=(OUT/'supervisor.lock')","lock=(W/'recovery_20261125/supervisor.lock')")
guard=guard.replace("if '--time-baseline' in sys.argv:","if True:")
guard=guard.replace("run='act_recovery_time'","run='act_progress_v1'").replace("W/'train_g2_act.py'","W/'train_progress_act.py'")
guard=guard.replace("'--steps','15000'","'--steps','8000'").replace("'--lr','2e-5'","'--lr','3e-5'")
guard=guard.replace("W/'runs/act_pilot4_absolute_refine/step_010000.pt'","W/'runs/act_recovery_time/step_015000.pt'")
guard=guard.replace("run_eval('act',W/'runs'/run/'step_015000.pt','recovery_act_time_15000'","run_eval('progress_act',W/'runs'/run/'step_008000.pt','act_progress_v1_8000'")
(W/'run_progress_guarded.py').write_text(guard)
names=['g2_progress_act_model.py','train_progress_act.py','progress_act_policy_server.py','run_progress_guarded.py']
for name in names:py_compile.compile(str(W/name),doraise=True)
(W/'progress_revision.json').write_text(json.dumps(dict(source_sha256={n:hashlib.sha256((W/n).read_bytes()).hexdigest() for n in names},changes=['Fourier progress embedding added to ACT state token','Training-only qpos dropout 50%, image dropout 30%','Transition-window oversampling','One shared checkpoint for four objects','Elapsed-time input; training-position overfit, not generalization'],physical_gates='unchanged'),indent=2))
print('PROGRESS_ACT_READY')
