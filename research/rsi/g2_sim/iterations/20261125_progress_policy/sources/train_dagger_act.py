from pathlib import Path
import argparse,json,random,time,os
import h5py,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from g2_dagger_act_model import make_model,normalize_images
from g2_force_features import encode_force
W=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--dataset',default='pilot4');p.add_argument('--run',default='act_pilot4');p.add_argument('--steps',type=int,default=15000);p.add_argument('--batch',type=int,default=4);p.add_argument('--chunk',type=int,default=50);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--kl',type=float,default=10.);p.add_argument('--resume');p.add_argument('--cosine',action='store_true');p.add_argument('--relative-arm',action='store_true');p.add_argument('--reset-action-head',action='store_true');p.add_argument('--phase-balanced',action='store_true');p.add_argument('--time-conditioned',action='store_true');p.add_argument('--deterministic-latent',action='store_true');p.add_argument('--state-skip',action='store_true');p.add_argument('--qpos-noise',type=float,default=0.);p.add_argument('--save-optimizer',action='store_true');a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
D=W/'datasets'/a.dataset;rows=json.loads((D/'manifest.json').read_text());stats=json.loads((D/'stats.json').read_text());O=W/'runs'/a.run;O.mkdir(parents=True,exist_ok=True)
import hashlib,shutil,platform
source_bundle=O/'source';source_bundle.mkdir(exist_ok=True);source_hashes={}
for source in [Path(__file__),W/'g2_act_model.py',W/'g2_progress_act_model.py',W/'g2_force_act_model.py',W/'g2_dagger_act_model.py',W/'g2_force_features.py',W/'vendor/act/detr/models/detr_vae.py',W/'vendor/act/detr/models/backbone.py',W/'vendor/act/detr/models/transformer.py']:
 shutil.copy2(source,source_bundle/source.name);source_hashes[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
(O/'provenance.json').write_text(json.dumps(dict(source_sha256=source_hashes,python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,seed=42,dataset=str(D)),indent=2))
if a.relative_arm:
 residuals=[]
 for row in rows:
  if row['split']!='train':continue
  with h5py.File(row['cache']) as f:
   q=f['qpos'][:];actions=f['action'][:];ts=np.arange(0,len(actions),5)
   for offset in [0,9,24,a.chunk-1]:
    valid=ts[ts+offset<len(actions)];values=actions[valid+offset].copy();values[:,:14]-=q[valid,:14];residuals.append(values)
 values=np.concatenate(residuals).astype(np.float64);stats['action_mean']=values.mean(0).tolist();stats['action_std']=np.maximum(values.std(0),.01).tolist();stats['action_representation']='arm_delta_from_current_qpos_gripper_absolute';stats['action_q01']=np.quantile(values,.01,axis=0).tolist();stats['action_q99']=np.quantile(values,.99,axis=0).tolist()
(O/'config.json').write_text(json.dumps(vars(a),indent=2));(O/'stats.json').write_text(json.dumps(stats,indent=2));(O/'manifest.json').write_text(json.dumps(rows,indent=2))
class Data(Dataset):
 def __init__(self,rows,validation=False):self.rows=rows;self.validation=validation;self.handles={};self.contacts={};self.groups=[[r for r in rows if r['task_id']==task] for task in dict.fromkeys(r['task_id'] for r in rows)]
 def __len__(self):return 1000000 if not self.validation else len(self.rows)*12
 def __getitem__(self,i):
  group=self.groups[i%len(self.groups)];row=self.rows[i%len(self.rows)] if self.validation else group[np.random.randint(len(group))]
  if row['cache'] not in self.handles:
   self.handles[row['cache']]=h5py.File(row['cache'],'r');phases=self.handles[row['cache']]['phase'].asstr()[:];self.contacts[row['cache']]=np.where(np.isin(phases,['close','contact_refine','contact_settle']))[0]
  f=self.handles[row['cache']];n=len(f['action']);t=min(n-1,int((i//len(self.rows)+.5)/12*n)) if self.validation else np.random.randint(n)
  if a.phase_balanced and not self.validation:
   band=np.random.rand();contact=self.contacts[row['cache']]
   if band<.10:t=np.random.randint(min(20,n))
   elif band<.30 and len(contact):t=int(np.random.choice(contact))
   elif band<.65:
    phases=f['phase'].asstr()[:];boundaries=np.flatnonzero(phases[1:]!=phases[:-1])+1
    if len(boundaries):t=int(np.clip(np.random.choice(boundaries)+np.random.randint(-25,51),0,n-1))
  raw_q=f['qpos'][t].copy()
  if a.qpos_noise and not self.validation:raw_q[:14]+=np.clip(np.random.normal(0,a.qpos_noise,14),-3*a.qpos_noise,3*a.qpos_noise)
  q=(raw_q-stats['qpos_mean'])/stats['qpos_std']
  if a.time_conditioned:
   progress=float(f['progress'][t]) if 'progress' in f else t/200.
   q=np.r_[q,encode_force(f['finger_forces'][t]),min(progress,stats['progress_terminal_limits'][row['asset']])]
  act=f['action'][t:min(t+a.chunk,n)]
  if a.relative_arm:act=act.copy();act[:,:14]-=raw_q[:14]
  count=len(act);actions=np.zeros((a.chunk,16),np.float32);actions[:count]=(act-stats['action_mean'])/stats['action_std'];mask=np.arange(a.chunk)>=count
  images=np.stack([f[name][t] for name in stats['camera_names']]).transpose(0,3,1,2).astype(np.float32)/255
  return torch.from_numpy(images),torch.tensor(q,dtype=torch.float32),torch.from_numpy(actions),torch.from_numpy(mask),torch.tensor(row['task_id'])
train_rows=[r for r in rows if r['split']=='train'];val_rows=[r for r in rows if r['split']=='validation']
loader=DataLoader(Data(train_rows),batch_size=a.batch,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True);iterator=iter(loader)
model=make_model(a.chunk,state_skip=a.state_skip,time_conditioned=a.time_conditioned).cuda();optim=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if 'backbones' not in n and 'state_action_skip' not in n]},{'params':[p for n,p in model.named_parameters() if 'backbones' in n],'lr':a.lr*.1}],lr=a.lr,weight_decay=1e-4)
if a.state_skip:optim.add_param_group({'params':list(model.state_action_skip.parameters()),'lr':a.lr*50})
if a.resume:
 loaded=torch.load(a.resume,map_location='cpu',weights_only=False)['model']
 if a.time_conditioned:
  initialized=model.state_dict()
  for key in ['input_proj_robot_state.weight','encoder_joint_proj.weight']:
   if loaded[key].shape[1]==17:
    expanded=initialized[key].cpu().clone();expanded[:,:16]=loaded[key][:,:16];expanded[:,-1]=loaded[key][:,-1];loaded[key]=expanded
 incompatible=model.load_state_dict(loaded,strict=False)
 assert not incompatible.unexpected_keys and all(k.startswith('force_encoder.') for k in incompatible.missing_keys),incompatible
if a.reset_action_head:
 model.action_head.reset_parameters();model.encoder_action_proj.reset_parameters()
def save(step):
 tmp=O/'latest.tmp.pt';torch.save(dict(model=model.state_dict(),step=step,stats=stats,config=vars(a),**({'optimizer':optim.state_dict()} if a.save_optimizer else {})),tmp);tmp.replace(O/'latest.pt')
 if step%1000==0 or step==0:
  retained=O/f'step_{step:06d}.pt'
  if retained.exists():retained.unlink()
  os.link(O/'latest.pt',retained)
def evaluate(step,subset,label):
 model.eval();errors=[]
 with torch.inference_mode():
  for images,q,actions,mask,task in DataLoader(Data(subset,True),batch_size=a.batch,num_workers=0):
   images,q,actions,mask,task=[x.cuda(non_blocking=True) for x in [images,q,actions,mask,task]]
   pred=model(q,normalize_images(images),None,task_ids=task)[0];err=(pred-actions).abs()*pred.new_tensor(stats['action_std']);valid=(~mask).unsqueeze(-1);errors.append(((err*valid).sum().item(),valid.sum().item()*16))
 metric=dict(step=step,split=label,action_mae_rad=sum(e[0] for e in errors)/sum(e[1] for e in errors));print(json.dumps(metric),flush=True)
 with (O/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
 model.train()
save(0);evaluate(0,train_rows,'train_zero_latent')
started=time.time();model.train()
for step in range(1,a.steps+1):
 if a.cosine:
  multiplier=.1+.9*.5*(1+np.cos(np.pi*step/a.steps))
  for group,scale in zip(optim.param_groups,[1.,.1,50.]):group['lr']=a.lr*scale*multiplier
 images,q,actions,mask,task=[x.cuda(non_blocking=True) for x in next(iterator)];optim.zero_grad(set_to_none=True)
 if a.deterministic_latent:
  pred=model(q,normalize_images(images),None,task_ids=task)[0];kl=pred.new_zeros(())
 else:
  pred,_,(mu,logvar)=model(q,normalize_images(images),None,actions,mask,task_ids=task);kl=(-.5*(1+logvar-mu.square()-logvar.exp()).sum(1)).mean()
 valid=(~mask).unsqueeze(-1);l1=((pred-actions).abs()*valid).sum()/(valid.sum()*16);loss=l1+a.kl*kl
 if not torch.isfinite(loss):raise RuntimeError('Nonfinite ACT loss')
 loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
 if step%25==0:
  metric=dict(step=step,loss=loss.item(),l1=l1.item(),kl=kl.item(),grad=float(grad),elapsed_s=time.time()-started);print(json.dumps(metric),flush=True)
  with (O/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
 if step%500==0:
  save(step);evaluate(step,train_rows,'train_zero_latent')
  if val_rows:evaluate(step,val_rows,'validation_zero_latent')
save(a.steps);print('ACT_TRAINING_COMPLETE',flush=True)





