"""CONTROL-CONTRACT DIAGNOSTIC ONLY. Not a learned policy or success evidence."""
from pathlib import Path
import json,hashlib,threading,subprocess,sys
from http.server import BaseHTTPRequestHandler,HTTPServer
import h5py,numpy as np
W=Path(__file__).resolve().parent
rows=json.loads((W/'datasets/pilot4/manifest.json').read_text());actions={}
for row in rows:
 if row['asset'] not in actions:
  with h5py.File(row['cache']) as f:actions[row['asset']]=f['action'][:]
sha=hashlib.sha256(b''.join(a.tobytes() for a in actions.values())).hexdigest();cursor=0
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_POST(self):
  global cursor
  request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  if self.path=='/load':
   cursor=0;response=dict(checkpoint='EXPERT_COMMAND_REPLAY_DIAGNOSTIC',sha256=sha,step=None,model='NOT_A_POLICY_EXPERT_REPLAY',expert_feedback=True)
  else:
   a=actions[request['asset']];chunk=a[np.minimum(np.arange(cursor,cursor+50),len(a)-1)];cursor+=10;response=dict(actions=chunk.tolist(),checkpoint_sha256=sha)
  data=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
server=HTTPServer(('127.0.0.1',19027),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
subprocess.run([sys.executable,str(W/'evaluate_g2_policy.py'),'--checkpoint','EXPERT_COMMAND_REPLAY_DIAGNOSTIC','--run','diagnostic_expert_command_replay','--port','19027','--assets','box_white','--evaluation-label','Expert command replay diagnostic ONLY; not learned policy'],check=True,cwd=W)
server.shutdown()
