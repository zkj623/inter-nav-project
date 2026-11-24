from pathlib import Path
import argparse,base64,hashlib,json,time
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np,torch
from g2_act_model import make_model,normalize_images
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=19024);a=p.parse_args();torch.set_num_threads(2)
model=None;stats=None;info={}
def load(path):
 global model,stats,info
 path=Path(path).resolve();assert path.is_relative_to(Path(__file__).resolve().parent/'runs')
 if info.get('checkpoint')==str(path):return info
 state=torch.load(path,map_location='cpu',weights_only=False);model=make_model(state['config']['chunk'],state_skip=state['config'].get('state_skip',False)).cuda();model.load_state_dict(state['model']);model.eval();stats=state['stats']
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 info=dict(checkpoint=str(path),sha256=h.hexdigest(),step=state['step'],model='ACT',chunk=state['config']['chunk'],inputs='head RGB,left wrist RGB,right wrist RGB,qpos16,task_id',expert_feedback=False)
 print('MODEL_LOADED',json.dumps(info),flush=True);return info
class Handler(BaseHTTPRequestHandler):
 def do_POST(self):
  try:
   request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   if self.path=='/load':response=load(request['checkpoint'])
   elif self.path=='/predict':
    assert model is not None
    if request.get('checkpoint_sha256'):assert request['checkpoint_sha256']==info['sha256']
    images=np.frombuffer(base64.b64decode(request['images']),np.uint8).reshape(3,224,224,3).copy();q=(np.array(request['qpos'])-stats['qpos_mean'])/stats['qpos_std'];task=stats['assets'].index(request['asset'])
    with torch.inference_mode():
     im=torch.from_numpy(images).permute(0,3,1,2).unsqueeze(0).cuda().float()/255.;qt=torch.tensor(q,dtype=torch.float32,device='cuda').unsqueeze(0);tid=torch.tensor([task],device='cuda');pred=model(qt,normalize_images(im),None,task_ids=tid)[0][0].cpu().numpy()
    actions=pred*np.array(stats['action_std'])+np.array(stats['action_mean'])
    if stats.get('action_representation')=='arm_delta_from_current_qpos_gripper_absolute':actions[:,:14]+=np.array(request['qpos'])[:14]
    assert np.isfinite(actions).all();response=dict(actions=actions.tolist(),checkpoint_sha256=info['sha256'])
   else:raise ValueError('Unknown endpoint')
   payload=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload)
  except Exception as ex:
   import traceback;traceback.print_exc();self.send_response(500);self.end_headers();self.wfile.write(json.dumps(dict(error=repr(ex))).encode())
 def log_message(self,*args):pass
print('ACT_POLICY_SERVER_READY',a.port,flush=True);HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

