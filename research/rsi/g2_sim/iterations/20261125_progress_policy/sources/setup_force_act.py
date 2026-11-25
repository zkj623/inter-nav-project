"""Prepare a causal simulated-contact-input ACT ablation; never starts a job."""
from pathlib import Path
import hashlib,json,py_compile
import h5py,numpy as np
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124'
common='''import numpy as np
FINGER_NAMES=[f'gripper_{s}_{f}_link4' for s in ('l','r') for f in ('inner','outer')]
def causal_force(samples,now):
 past=[s for s in samples if float(s['physics_time_s'])<=now]
 if not past:raise ValueError('No past contact measurement')
 selected=past[-3:];stamp=float(selected[-1]['physics_time_s'])
 if not 0<=now-stamp<=.101:raise ValueError('Stale or future contact measurement')
 values=np.array([[s['finger_forces'][n] for n in FINGER_NAMES] for s in selected],dtype=np.float64)
 if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid contact measurement')
 return values.mean(0),stamp
def encode_force(force):
 values=np.asarray(force)
 if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid contact measurement')
 return np.log1p(np.minimum(values,100.))/np.log(36.)
'''
(W/'g2_force_features.py').write_text(common)
(R/'g2_force_features.py').write_text(common)
from g2_force_features import causal_force,FINGER_NAMES
D=W/'datasets/pilot4_force_causal';D.mkdir(exist_ok=True)
rows=json.loads((W/'datasets/pilot4_one_each/manifest.json').read_text());audit=[]
for row in rows:
 trial=Path(row['trial']);samples=json.loads((trial/'samples.json').read_text());cache=Path(row['cache']);destination=D/(row['asset']+'.hdf5')
 assert not destination.exists(),destination
 with h5py.File(trial/'episode.hdf5') as raw,h5py.File(cache) as src,h5py.File(destination,'w') as dst:
  times=raw['timestamps'][:];assert len(times)==len(src['action'])
  assert np.array_equal(raw['observations/qpos'][:],src['qpos'][:])
  values=[causal_force(samples,float(t)) for t in times]
  forces=np.stack([v[0] for v in values]);stamps=np.array([v[1] for v in values])
  for key in src:dst[key]=h5py.ExternalLink(str(cache),'/'+key)
  dst['finger_forces']=forces.astype('f4');dst['force_timestamps']=stamps;dst['timestamps']=times
  dst.attrs['force_semantics']='Latest three past 10Hz measurements averaged; four distal finger-object contact force norms, simulator only'
  dst.attrs['force_names']=json.dumps(FINGER_NAMES)
  audit.append(dict(asset=row['asset'],frames=len(times),minimum_age_s=float((times-stamps).min()),maximum_age_s=float((times-stamps).max()),no_future_measurements=bool((times>=stamps).all()),source_samples_sha256=hashlib.sha256((trial/'samples.json').read_bytes()).hexdigest(),source_episode_sha256=row['source_sha256']))
 row['cache']=str(destination)
(D/'manifest.json').write_text(json.dumps(rows,indent=2));stats=json.loads((W/'datasets/pilot4_one_each/stats.json').read_text());stats['force_names']=FINGER_NAMES;stats['force_semantics']='Simulator object-filtered contact forces; hardware sensor availability unverified'
(D/'stats.json').write_text(json.dumps(stats,indent=2));(D/'causal_alignment_audit.json').write_text(json.dumps(audit,indent=2))
model='''import torch
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
'''
(W/'g2_force_act_model.py').write_text(model)
src=(W/'train_progress_act.py').read_text(encoding='utf-8-sig').replace('from g2_progress_act_model import','from g2_force_act_model import')
src=src.replace('W=Path(__file__).resolve().parent','from g2_force_features import encode_force\nW=Path(__file__).resolve().parent')
src=src.replace("W/'g2_progress_act_model.py',","W/'g2_progress_act_model.py',W/'g2_force_act_model.py',W/'g2_force_features.py',")
src=src.replace("if a.time_conditioned:q=np.r_[q,t/200.]","if a.time_conditioned:q=np.r_[q,encode_force(f['finger_forces'][t]),t/200.]")
old="if loaded[key].shape[1]==16:\n    expanded=initialized[key].cpu().clone();expanded[:,:16]=loaded[key];loaded[key]=expanded"
new="if loaded[key].shape[1]==17:\n    expanded=initialized[key].cpu().clone();expanded[:,:16]=loaded[key][:,:16];expanded[:,-1]=loaded[key][:,-1];loaded[key]=expanded"
assert old in src;src=src.replace(old,new).replace("k.startswith('progress_encoder.')","k.startswith('force_encoder.')")
(W/'train_force_act.py').write_text(src)
server=(W/'progress_act_policy_server.py').read_text().replace('from g2_progress_act_model import','from g2_force_act_model import')
server=server.replace('p=argparse.ArgumentParser();','from g2_force_features import encode_force\np=argparse.ArgumentParser();')
server=server.replace('q=np.r_[q,min(',"q=np.r_[q,encode_force(request['finger_forces']),min(")
server=server.replace("'ACT_progress_conditioned'","'ACT_progress_sim_contact'").replace('qpos16,task_id','qpos16,task_id,elapsed time,causal averaged four simulated distal contact forces')
(W/'force_act_policy_server.py').write_text(server)
adapter='''from pathlib import Path
import hashlib,numpy as np
from policy_time_recorder import Recorder as Base
from g2_force_features import causal_force
class Recorder(Base):
 def set_contact_observation(self,samples,now):
  self.contact_force,self.contact_stamp=causal_force(samples,now)
  self.contact_now=now
 def post(self,path,value):
  if path=='/predict':value={**value,'finger_forces':self.contact_force.tolist()}
  return super().post(path,value)
 def command(self,*args,**kwargs):
  previous=self.count;held=super().command(*args,**kwargs)
  if self.count>previous:
   for name,val in [('observations/finger_forces',self.contact_force),('force_timestamps',self.contact_stamp)]:
    val=np.asarray(val);f=self.file
    if name not in f:f.create_dataset(name,shape=(0,*val.shape),maxshape=(None,*val.shape),dtype='f8')
    f[name].resize(self.count,axis=0);f[name][self.count-1]=val
   assert abs(float(self.file['timestamps'][self.count-1])-self.contact_now)<1e-8
  return held
 def finish(self,result):
  super().finish(result)
  result['policy_evaluation']['force_adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  result['policy_evaluation']['force_features_sha256']=hashlib.sha256((Path(__file__).parent/'g2_force_features.py').read_bytes()).hexdigest()
  result['policy_evaluation']['control_scope']='Learned ACT: RGB, qpos, task, elapsed time, latest three past simulated distal finger contact norms; original joint limits, no expert targets or scripted force-balancing controller'
'''
(R/'policy_force_recorder.py').write_text(adapter)
for row in rows:
 asset=row['asset'];source=(R/f'policy_time_{asset}_trial.py').read_text(encoding='utf-8-sig')
 source=source.replace('from policy_time_recorder import Recorder','from policy_force_recorder import Recorder')
 old='        target=recorder.command(target,phase,robot,world)'
 assert source.count(old)==1
 source=source.replace(old,"        if samples:recorder.set_contact_observation(samples,float(world.current_time))\n"+old)
 trial_source=R/f'policy_force_{asset}_trial.py';trial_source.write_text(source)
 source_data=trial_source.read_bytes();source_sha=hashlib.sha256(source_data).hexdigest();archive=R/'source_versions'/(source_sha+'.py');archive.parent.mkdir(exist_ok=True)
 if archive.exists():assert archive.read_bytes()==source_data
 else:archive.write_bytes(source_data)
evaluation=(W/'evaluate_g2_policy.py').read_text(encoding='utf-8-sig').replace("f'policy_time_{asset}_trial.py'","f'policy_force_{asset}_trial.py'")
(W/'evaluate_force_policy.py').write_text(evaluation)
guard=(W/'run_progress_guarded.py').read_text().replace("OUT=W/'progress_act_20261125'","OUT=W/'force_act_20261125'")
guard=guard.replace("W/'evaluate_g2_policy.py'","W/'evaluate_force_policy.py'")
guard=guard.replace('Four demonstrations, elapsed-time conditioned ACT overfit; no generalization claim','Four demonstrations; learned ACT with causal simulated contact forces and time; no hardware or generalization claim')
guard=guard.replace("run='act_progress_v1'","run='act_force_v1'").replace("W/'train_progress_act.py'","W/'train_force_act.py'")
guard=guard.replace("'pilot4_one_each'","'pilot4_force_causal'").replace("W/'runs/act_recovery_time/step_015000.pt'","W/'runs/act_progress_v1/step_008000.pt'")
guard=guard.replace("run_eval('progress_act'","run_eval('force_act'").replace("'act_progress_v1_8000'","'act_force_v1_8000'")
# 20Hz target updates, replan every two frames: 10Hz input force updates.
guard=guard.replace("'--port',str(port)]+extra","'--port',str(port),'--execute','2']+extra")
(W/'run_force_guarded.py').write_text(guard)
names=['g2_force_features.py','g2_force_act_model.py','train_force_act.py','force_act_policy_server.py','evaluate_force_policy.py','run_force_guarded.py']
for name in names:py_compile.compile(str(W/name),doraise=True)
for path in [R/'policy_force_recorder.py',*[R/f"policy_force_{r['asset']}_trial.py" for r in rows]]:py_compile.compile(str(path),doraise=True)
(D/'revision.json').write_text(json.dumps(dict(files={n:hashlib.sha256((W/n).read_bytes()).hexdigest() for n in names},scope='Additional causal simulator contact observation only; unchanged physical acceptance; no newly collected demonstrations'),indent=2))
print('FORCE_ABLATION_PREPARED_NOT_STARTED')
