from pathlib import Path
import py_compile,json
W=Path(__file__).resolve().parent
(W/'g2_timeonly_act_model.py').write_text('''import torch
from g2_progress_act_model import make_model as progress_model,normalize_images
def make_model(*args,**kwargs):
 model=progress_model(*args,**kwargs)
 def mask(module,inputs):
  q,image,*other=inputs;q=q.clone();q[:,:16]=0
  return (q,torch.zeros_like(image),*other)
 model.register_forward_pre_hook(mask)
 return model
''')
s=(W/'train_progress_act.py').read_text().replace('from g2_progress_act_model import','from g2_timeonly_act_model import')
s=s.replace("W/'g2_progress_act_model.py',","W/'g2_progress_act_model.py',W/'g2_timeonly_act_model.py',")
(W/'train_timeonly_act.py').write_text(s)
s=(W/'progress_act_policy_server.py').read_text().replace('from g2_progress_act_model import','from g2_timeonly_act_model import').replace('ACT_progress_conditioned','ACT_time_only_ablation')
s=s.replace('head RGB,left wrist RGB,right wrist RGB,qpos16,task_id','task_id and elapsed_time; RGB and qpos masked by explicit ablation')
s=s.replace("info['terminal_progress_limits']=", "info['observation_ablation']='Learned task/time-only ACT overfit baseline; no visual or joint-state feedback, no expert replay';info['terminal_progress_limits']=")
(W/'timeonly_act_policy_server.py').write_text(s)
s=(W/'run_progress_guarded.py').read_text().replace("OUT=W/'progress_act_20261125'","OUT=W/'timeonly_act_20261125'")
s=s.replace("run='act_progress_v1'","run='act_timeonly_v2'").replace("W/'train_progress_act.py'","W/'train_timeonly_act.py'")
s=s.replace("W/'runs/act_recovery_time/step_015000.pt'","W/'runs/act_progress_v1/step_008000.pt'")
s=s.replace("run_eval('progress_act'","run_eval('timeonly_act'").replace("'act_progress_v1_8000'","'act_timeonly_v2_8000'")
s=s.replace('Four demonstrations, elapsed-time conditioned ACT overfit; no generalization claim','Four demonstrations, task/time-only ACT input ablation; no visual or qpos feedback; no generalization claim')
(W/'run_timeonly_guarded.py').write_text(s)
for n in ['g2_timeonly_act_model.py','train_timeonly_act.py','timeonly_act_policy_server.py','run_timeonly_guarded.py']:py_compile.compile(str(W/n),doraise=True)
print('TIME_ONLY_ABLATION_PREPARED_NOT_STARTED')
