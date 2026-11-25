"""Fine-tune official pi05 with LoRA on accepted G2 observations and actions."""
from pathlib import Path
import argparse,sys,json,time,dataclasses
import h5py,numpy as np
W=Path(__file__).resolve().parent;P=W/'vendor/openpi';sys.path[:0]=[str(P),str(P/'src')]
import jax
import flax.nnx as nnx
from openpi.models import pi0_config,model,tokenizer
from openpi.training import config,optimizer,weight_loaders
from openpi.shared import nnx_utils,normalize
from scripts import train
p=argparse.ArgumentParser();p.add_argument('--dataset',default='pilot4');p.add_argument('--steps',type=int,default=5000);p.add_argument('--batch',type=int,default=1);p.add_argument('--chunk',type=int,default=50);p.add_argument('--run',default='pilot4');p.add_argument('--init-checkpoint');p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--checkpoint-root',default='/tmp/zkj_g2_20261125_checkpoints');a=p.parse_args()
D=W/'datasets'/a.dataset
while not (D/'manifest.json').exists() or not (W/'weights/pi05_base/DOWNLOAD_COMPLETE').exists():time.sleep(15)
rows=[r for r in json.loads((D/'manifest.json').read_text()) if r['split']=='train'];stats=json.loads((D/'stats.json').read_text());tok=tokenizer.PaligemmaTokenizer(200)
def bounds(key):
 low=np.array(stats[key+'_q01'],np.float32);high=np.maximum(np.array(stats[key+'_q99'],np.float32),low+.02);return low,high
residual_parts=[]
for row in rows:
 with h5py.File(row['cache']) as f:
  q=f['qpos'][:];act=f['action'][:];starts=np.arange(0,len(act),5)
  indices=np.minimum(starts[:,None]+np.arange(a.chunk),len(act)-1)
  values=act[indices].copy();values[:,:,:14]-=q[starts,None,:14];residual_parts.append(values.reshape(-1,16))
values=np.concatenate(residual_parts).astype(np.float64)
for key,value in [('mean',values.mean(0)),('std',np.maximum(values.std(0),.01)),('q01',np.quantile(values,.01,axis=0)),('q99',np.quantile(values,.99,axis=0))]:stats['action_'+key]=value.tolist()
del values,residual_parts
ql,qh=bounds('qpos');al,ah=bounds('action')
norm_stats={key:normalize.NormStats(mean=np.pad(np.array(stats[src+'_mean']), (0,16)),std=np.pad(np.array(stats[src+'_std']),(0,16),constant_values=1),q01=np.pad(lo,(0,16),constant_values=-1),q99=np.pad(hi,(0,16),constant_values=1)) for key,src,lo,hi in [('state','qpos',ql,qh),('actions','action',al,ah)]}
checkpoint_dir=Path(a.checkpoint_root)/'g2_pi05_lora'/a.run
completed_steps=[int(p.name)+1 for p in checkpoint_dir.glob('*') if p.name.isdigit() and (p/'_CHECKPOINT_METADATA').exists() and (p/'params').exists()]
data_seed=42+max(completed_steps,default=0)
class Loader:
 def __init__(self,sharding):self.sharding=sharding;self.rng=np.random.default_rng(data_seed);self.handles={};self.groups=[[r for r in rows if r['asset']==asset] for asset in dict.fromkeys(r['asset'] for r in rows)]
 def data_config(self):return config.DataConfig(asset_id='g2',norm_stats=norm_stats,use_quantile_norm=True)
 def __iter__(self):
  while True:
   images={k:[] for k in model.IMAGE_KEYS};states=[];tokens=[];masks=[];actions=[]
   for _ in range(a.batch):
    group=rows if a.dataset=='pilot4' else self.groups[int(self.rng.integers(len(self.groups)))]
    row=group[int(self.rng.integers(len(group)))];path=row['cache']
    if path not in self.handles:self.handles[path]=h5py.File(path,'r')
    f=self.handles[path];n=len(f['action']);t=int(self.rng.integers(n));q=np.pad((f['qpos'][t]-ql)/(qh-ql)*2-1,(0,16)).astype(np.float32)
    text='Lift the '+row['asset'].replace('_',' ')+' with both arms and hold it above the floor.'
    token,mask=tok.tokenize(text,np.clip(q,-1,1));tokens.append(token.astype(np.int32));masks.append(mask);states.append(q)
    action=f['action'][t:min(n,t+a.chunk)].copy();action[:,:14]-=f['qpos'][t,:14];action=np.pad(action,((0,a.chunk-len(action)),(0,0)),mode='edge');action=(action-al)/(ah-al)*2-1;actions.append(np.pad(action,((0,0),(0,16))).astype(np.float32))
    for key,name in zip(model.IMAGE_KEYS,stats['camera_names']):images[key].append(f[name][t].astype(np.float32)/127.5-1.)
   obs=model.Observation(images={k:np.stack(v) for k,v in images.items()},image_masks={k:np.ones(a.batch,bool) for k in images},state=np.stack(states),tokenized_prompt=np.stack(tokens),tokenized_prompt_mask=np.stack(masks))
   yield jax.tree.map(lambda x:jax.device_put(x,self.sharding),(obs,np.stack(actions)))
mc=pi0_config.Pi0Config(pi05=True,action_horizon=a.chunk,paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
cfg=config.TrainConfig(name='g2_pi05_lora',exp_name=a.run,model=mc,weight_loader=weight_loaders.CheckpointWeightLoader(a.init_checkpoint or str(W/'weights/pi05_base/params')),freeze_filter=nnx.Any(mc.get_freeze_filter(),nnx_utils.PathRegex('.*img.*')),batch_size=a.batch,num_train_steps=a.steps,ema_decay=None,lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100,peak_lr=a.lr,decay_steps=36000,decay_lr=a.lr*.1),optimizer=optimizer.AdamW(clip_gradient_norm=1.),checkpoint_base_dir=a.checkpoint_root,assets_base_dir=str(W/'pi05_assets'),save_interval=1000000,keep_period=None,log_interval=10,wandb_enabled=False,resume=True)
train._data_loader.create_data_loader=lambda configuration,sharding,shuffle:Loader(sharding)
scope=dict(model='pi05 official pretrained base with LoRA on language and action expert; frozen visual backbone',dataset=str(D),assets=sorted(set(r['asset'] for r in rows)),episodes=len(rows),state_dim=16,padded_model_dim=32,action_representation='arm_delta_from_current_qpos_gripper_absolute',action_semantics='14 arm targets relative to current measured qpos;2 absolute grippers;20Hz',action_horizon=a.chunk,image_processing=stats['image_processing'],quantile_low_state=ql.tolist(),quantile_high_state=qh.tolist(),quantile_low_action=al.tolist(),quantile_high_action=ah.tolist(),training=dict(steps=a.steps,batch=a.batch,lr=a.lr,data_seed=data_seed,init_checkpoint=a.init_checkpoint,sampling=('uniform episode' if a.dataset=='pilot4' else 'uniform object then episode')+' and start time; original openpi visual augmentation'),task_prompt='Lift the <asset name> with both arms and hold it above the floor.')
(W/('pi05_'+a.run+'_scope.json')).write_text(json.dumps(scope,indent=2));print('PI05_TRAINING_START',json.dumps(scope),flush=True)
# Avoid overlapping full-model writes to shared disks.
original_save=train._checkpoints.save_state
def synchronous_save(manager,*args,**kwargs):
 original_save(manager,*args,**kwargs)
 manager.wait_until_finished()
train._checkpoints.save_state=synchronous_save
(W/'pi05_focus_20261125'/('trainer_'+a.run+'.py')).write_text(Path(__file__).read_text())
original_init_wandb=train.init_wandb
def init_with_local_metrics(*args,**kwargs):
 result=original_init_wandb(*args,**kwargs)
 original_log=train.wandb.log
 def log_local(metrics,*args,**kwargs):
  numeric={k:float(v) for k,v in metrics.items() if isinstance(v,(int,float,np.number)) or (isinstance(v,np.ndarray) and v.ndim==0)}
  if numeric:
   if not all(np.isfinite(v) for v in numeric.values()):raise RuntimeError('NONFINITE_TRAINING_METRIC '+str(numeric))
   record=dict(step=kwargs.get('step'),time=time.strftime('%FT%T'),**numeric)
   with (W/'pi05_focus_20261125'/(a.run+'_metrics.jsonl')).open('a') as f:f.write(json.dumps(record)+'\n')
   if kwargs.get('step',0)%100==0:print('TRAIN_METRIC',json.dumps(record),flush=True)
  return original_log(metrics,*args,**kwargs)
 train.wandb.log=log_local
 return result
train.init_wandb=init_with_local_metrics
train.main(cfg)
print('FOCUS_TRAINING_COMPLETE',flush=True)
