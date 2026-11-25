from pathlib import Path
import json,h5py,hashlib,py_compile
W=Path(__file__).resolve().parent
rows=json.loads((W/'datasets/pilot4_one_each/manifest.json').read_text());limits={}
for r in rows:
 with h5py.File(r['cache']) as f:
  assert all(p=='hold' for p in f['phase'].asstr()[-50:])
  limits[r['asset']]=(len(f['action'])-50)/200.
scope=W/'progress_terminal_limits.json';scope.write_text(json.dumps(limits,indent=2))
p=W/'progress_act_policy_server.py';src=p.read_text()
src=src.replace("model=None;stats=None;info={};time_conditioned=False","model=None;stats=None;info={};time_conditioned=False\nterminal_limits=json.loads((Path(__file__).resolve().parent/'progress_terminal_limits.json').read_text())")
src=src.replace("float(request['elapsed_s'])/10.","min(float(request['elapsed_s'])/10.,terminal_limits[request['asset']])")
src=src.replace("print('MODEL_LOADED'","info['terminal_progress_limits']=terminal_limits;info['terminal_progress_scope']='Clamp elapsed progress to trained final hold; no reference actions or object feedback';print('MODEL_LOADED'")
assert 'min(float' in src and 'terminal_progress_limits' in src
p.write_text(src);py_compile.compile(str(p),doraise=True)
print(json.dumps(dict(limits_scaled_seconds_div10=limits,server_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
