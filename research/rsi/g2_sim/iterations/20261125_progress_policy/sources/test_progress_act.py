from pathlib import Path
import torch,json
from g2_progress_act_model import make_model
torch.set_num_threads(2);torch.manual_seed(42);W=Path(__file__).resolve().parent
s=torch.load(W/'runs/act_recovery_time/step_015000.pt',map_location='cpu',weights_only=False,mmap=True)
m=make_model(time_conditioned=True);missing=m.load_state_dict(s['model'],strict=False)
assert not missing.unexpected_keys and all(x.startswith('progress_encoder.') for x in missing.missing_keys)
q=torch.zeros(2,17);q[1,-1]=4;im=torch.zeros(2,3,3,224,224);task=torch.tensor([7,7]);m.eval()
with torch.no_grad():a=m(q,im,None,task_ids=task)[0];b=m(q,im,None,task_ids=task)[0]
assert a.shape==(2,50,16) and torch.isfinite(a).all() and torch.equal(a,b)
effect=float(abs(a[0]-a[1]).mean());assert effect>1e-5
m.train();pred=m(q,im,None,task_ids=task)[0];pred.square().mean().backward()
grad=sum(float(p.grad.abs().sum()) for p in m.progress_encoder.parameters() if p.grad is not None);assert grad>0
report=dict(passed=True,shape=list(a.shape),deterministic_eval=True,time_effect=effect,progress_gradient=grad,missing_keys=missing.missing_keys)
(W/'progress_model_test.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
