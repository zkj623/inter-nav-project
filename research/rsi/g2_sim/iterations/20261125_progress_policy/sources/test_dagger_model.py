from pathlib import Path
import json,torch
from g2_force_act_model import make_model as old_model
from g2_dagger_act_model import make_model
W=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.manual_seed(9)
state=torch.load(W/'runs/act_force_v1/step_008000.pt',map_location='cpu',weights_only=False)['model']
old=old_model();old.load_state_dict(state);old.eval();new=make_model();new.load_state_dict(state);new.eval()
q=torch.randn(2,21);im=torch.randn(2,3,3,64,64);tasks=torch.tensor([3,7])
with torch.no_grad():
 a=old(q,im,None,task_ids=tasks)[0];b=new(q,im,None,task_ids=tasks)[0]
 assert torch.equal(a,b)
 del old,state
 new.train();c=new(q,im,None,task_ids=tasks)[0];d=new(q,im,None,task_ids=tasks)[0]
 assert torch.equal(c,d),'Unexpected training observation masking or stochastic inference'
out=dict(checkpoint_output_preserved=True,training_observations_unmasked=True,shape=list(b.shape),finite=bool(torch.isfinite(b).all()))
(W/'dagger_model_test.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
