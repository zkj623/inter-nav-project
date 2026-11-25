from pathlib import Path
import json
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124'
rows=json.loads((W/'datasets/pilot4_one_each/manifest.json').read_text())
fig,ax=plt.subplots(3,3,figsize=(15,9),sharex='col',constrained_layout=True)
finger_names=[f'gripper_{s}_{f}_link4' for s in ('l','r') for f in ('inner','outer')]
for col,asset in enumerate(['bin_lidded','bin_pedal','basket_rectangular']):
 row=next(r for r in rows if r['asset']==asset)
 for source,label,style in [(Path(row['trial']),'Expert','--'),(R/'policy_trials/act_progress_v1_8000'/row['plan_dir']/asset/'candidate0_repeat1','ACT','-')]:
  with h5py.File(source/'episode.hdf5' if (source/'episode.hdf5').exists() else source/'episode.partial.hdf5') as f:start=float(f['timestamps'][0])
  samples=json.loads((source/'samples.json').read_text());samples=[s for s in samples if s['physics_time_s']>=start]
  t=np.array([s['physics_time_s']-start for s in samples])
  ax[0,col].plot(t,[100*s['lift_m'] for s in samples],style,label=label)
  ax[1,col].plot(t,[s['tilt_deg'] for s in samples],style,label=label)
  if label=='ACT':
   for n,short in zip(finger_names,['L inner','L outer','R inner','R outer']):ax[2,col].plot(t,[s['finger_forces'][n] for s in samples],label=short,linewidth=1)
 ax[0,col].set_title(asset);ax[0,col].axhline(15,color='gray',linewidth=.7)
 ax[1,col].axhline(20,color='gray',linewidth=.7);ax[2,col].set_xlabel('Elapsed policy time (s)')
 for row_ax in ax[:,col]:row_ax.grid(alpha=.2);row_ax.legend(fontsize=8)
ax[0,0].set_ylabel('Lift (cm)');ax[1,0].set_ylabel('Tilt (degrees)');ax[2,0].set_ylabel('ACT distal contact force (N)')
fig.suptitle('Three failed ACT rollouts: lost inner contacts and unstable support\nSame 8000-step checkpoint, execute_steps=10; original physical acceptance unchanged')
out=W/'progress_contact_failures.png';fig.savefig(out,dpi=140);print(out)
