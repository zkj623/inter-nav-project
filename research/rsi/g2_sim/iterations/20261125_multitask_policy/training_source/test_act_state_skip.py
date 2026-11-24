import torch,json
from g2_act_model import make_model
torch.set_num_threads(2);torch.manual_seed(7)
m=make_model(50,state_skip=True).eval();q=torch.randn(1,16);im=torch.rand(1,3,3,224,224);task=torch.tensor([1])
with torch.inference_mode():
 zero=m(q,im,None,task_ids=task)[0]
 m.state_action_skip.weight.copy_(torch.eye(16))
 shifted=m(q,im,None,task_ids=task)[0]
 error=float((shifted-zero-q[:,None,:]).abs().max())
assert error<1e-6,error
print(json.dumps(dict(state_skip_forward_verified=True,max_error=error,shape=list(shifted.shape))))
