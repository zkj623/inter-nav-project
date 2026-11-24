from pathlib import Path
import argparse,json,sys
import h5py,numpy as np,torch
from g2_act_model import make_model,normalize_images
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';sys.path.insert(0,str(R));from g2_kinematics import Kinematics
p=argparse.ArgumentParser();p.add_argument('checkpoint');a=p.parse_args();state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);stats=state['stats'];model=make_model(state['config']['chunk']).cuda().eval();rows=json.loads((Path(a.checkpoint).parent/'manifest.json').read_text());k=Kinematics(R/'generated/g2_crs_omnipicker.urdf');measurements=[];torch.set_num_threads(2)
model.load_state_dict(state['model'])
with torch.inference_mode():
 for row in rows:
  result=json.loads((Path(row['trial'])/'result.json').read_text());plan=result['plan'];base=np.array(plan['base_pose']);tcp=np.array(plan['tcp'])
  with h5py.File(row['cache']) as f:
   n=len(f['action'])
   for t in np.unique(np.linspace(0,n-1,24,dtype=int)):
    im=np.stack([f[name][t] for name in stats['camera_names']]).transpose(0,3,1,2).astype(np.float32)/255;qt=(f['qpos'][t]-stats['qpos_mean'])/stats['qpos_std'];pred=model(torch.tensor(qt,dtype=torch.float32,device='cuda')[None],normalize_images(torch.from_numpy(im).cuda()[None]),None,task_ids=torch.tensor([row['task_id']],device='cuda'))[0][0].cpu().numpy();pred=pred*np.array(stats['action_std'])+stats['action_mean'];pred[:,:14]+=f['qpos'][t,:14] if stats.get('action_representation')=='arm_delta_from_current_qpos_gripper_absolute' else 0;count=min(len(pred),n-t);actual=f['action'][t:t+count];error=abs(pred[:count]-actual)
    target_q={**plan['q_start'],**dict(zip(stats['joint_names'],actual[0]))};pred_q={**plan['q_start'],**dict(zip(stats['joint_names'],pred[0]))};target_fk=k.fk(target_q,base);pred_fk=k.fk(pred_q,base);dist=[float(np.linalg.norm((target_fk[f'arm_{s}_end_link']@tcp)[:3,3]-(pred_fk[f'arm_{s}_end_link']@tcp)[:3,3])*1000) for s in ['l','r']]
    phase=f['phase'].asstr()[t];measurements.append(dict(asset=row['asset'],phase=phase,frame=int(t),first_action_mae_rad=float(error[0].mean()),first10_mae_rad=float(error[:10].mean()),chunk_mae_rad=float(error.mean()),first_arm_mae_rad=float(error[0,:14].mean()),first_gripper_mae_rad=float(error[0,14:].mean()),tcp_error_mm=float(np.mean(dist))))
keys=[k for k in measurements[0] if k.endswith('_rad') or k.endswith('_mm')];summary={asset:{key:float(np.mean([r[key] for r in measurements if r['asset']==asset])) for key in keys} for asset in dict.fromkeys(r['asset'] for r in measurements)}
out=dict(checkpoint=a.checkpoint,weights_loaded=True,step=state['step'],per_asset=summary,per_phase={phase:{key:float(np.mean([r[key] for r in measurements if r['phase']==phase])) for key in keys} for phase in dict.fromkeys(r['phase'] for r in measurements)},samples=measurements)
dest=Path(a.checkpoint).with_suffix('.analysis.json');dest.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='samples'}),flush=True)

