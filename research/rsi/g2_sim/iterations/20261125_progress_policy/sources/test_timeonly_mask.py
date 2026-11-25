from pathlib import Path
import ast,torch,json
W=Path(__file__).resolve().parent;torch.set_num_threads(1)
tree=ast.parse((W/'g2_timeonly_act_model.py').read_text());outer=next(x for x in tree.body if isinstance(x,ast.FunctionDef));mask=next(x for x in outer.body if isinstance(x,ast.FunctionDef));env={'torch':torch};exec(compile(ast.Module(body=[mask],type_ignores=[]),'<mask>','exec'),env)
q=torch.randn(2,17,requires_grad=True);images=torch.randn(2,3,3,8,8);before=q.detach().clone();other=object()
out=env['mask'](None,(q,images,other))
assert (out[0][:,:16]==0).all() and torch.equal(out[0][:,-1],q[:,-1]) and (out[1]==0).all() and out[2] is other and torch.equal(q.detach(),before)
out[0].sum().backward();assert (q.grad[:,:16]==0).all() and (q.grad[:,-1]==1).all()
report=dict(passed=True,observations_masked=True,time_preserved=True,no_input_mutation=True,time_gradients_preserved=True)
(W/'timeonly_mask_test.json').write_text(json.dumps(report));print(json.dumps(report))
