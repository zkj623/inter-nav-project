from pathlib import Path
import sys,argparse,json,base64,hashlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np
W=Path(__file__).resolve().parent;P=W/'vendor/openpi';sys.path[:0]=[str(P),str(P/'src')]
import jax,jax.numpy as jnp
from openpi.models import pi0_config,model as model_api,tokenizer
from openpi.shared import nnx_utils
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=19025);p.add_argument('--scope',default='pi05_pilot4_scope.json');a=p.parse_args();scope=json.loads((W/a.scope).read_text());tok=tokenizer.PaligemmaTokenizer(200)
ql=np.array(scope['quantile_low_state']);qh=np.array(scope['quantile_high_state']);al=np.array(scope['quantile_low_action']);ah=np.array(scope['quantile_high_action']);info={};sample=None;rng=jax.random.key(42)
def load(path):
 global sample,info
 path=Path(path).resolve();assert path.is_relative_to(W/'checkpoints') or path.is_relative_to(Path('/tmp/zkj_g2_20261125_checkpoints')) or path.is_relative_to(W/'checkpoint_backups')
 if info.get('checkpoint')==str(path):return info
 cfg=pi0_config.Pi0Config(pi05=True,action_horizon=scope['action_horizon'],paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
 policy=cfg.load(model_api.restore_params(path/'params',dtype=jnp.bfloat16));sample=nnx_utils.module_jit(policy.sample_actions)
 h=hashlib.sha256()
 for f in sorted((path/'params').rglob('*')):
  if f.is_file():
   h.update(str(f.relative_to(path)).encode())
   with f.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 info=dict(checkpoint=str(path),sha256=h.hexdigest(),step=int(path.name),model='pi05_LoRA',chunk=scope['action_horizon'],inputs='head RGB,left wrist RGB,right wrist RGB,qpos16,text task',expert_feedback=False);print('MODEL_LOADED',json.dumps(info),flush=True);return info
class Handler(BaseHTTPRequestHandler):
 def do_POST(self):
  global rng
  try:
   request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   if self.path=='/load':response=load(request['checkpoint'])
   elif self.path=='/predict':
    assert sample is not None and request['checkpoint_sha256']==info['sha256'];images=np.frombuffer(base64.b64decode(request['images']),np.uint8).reshape(3,224,224,3);q=np.pad((np.array(request['qpos'])-ql)/(qh-ql)*2-1,(0,16)).astype(np.float32);text='Lift the '+request['asset'].replace('_',' ')+' with both arms and hold it above the floor.';token,mask=tok.tokenize(text,np.clip(q,-1,1))
    obs=model_api.Observation(images={name:jnp.asarray(images[i][None].astype(np.float32)/127.5-1) for i,name in enumerate(model_api.IMAGE_KEYS)},image_masks={name:jnp.ones((1,),bool) for name in model_api.IMAGE_KEYS},state=jnp.asarray(q[None]),tokenized_prompt=jnp.asarray(token[None],dtype=jnp.int32),tokenized_prompt_mask=jnp.asarray(mask[None]))
    rng,key=jax.random.split(rng);pred=np.asarray(sample(key,obs,num_steps=10))[0,:,:16];actions=(pred+1)/2*(ah-al)+al;actions[:,:14]+=np.array(request['qpos'])[:14] if scope.get('action_representation')=='arm_delta_from_current_qpos_gripper_absolute' else 0;assert np.isfinite(actions).all();response=dict(actions=actions.tolist(),checkpoint_sha256=info['sha256'])
   else:raise ValueError('Unknown endpoint')
   self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(response).encode())
  except Exception as ex:
   import traceback;traceback.print_exc();self.send_response(500);self.end_headers();self.wfile.write(json.dumps(dict(error=repr(ex))).encode())
 def log_message(self,*args):pass
print('PI05_POLICY_SERVER_READY',a.port,flush=True);HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

