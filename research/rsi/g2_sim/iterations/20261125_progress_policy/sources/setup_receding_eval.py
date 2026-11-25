from pathlib import Path
import os,signal,py_compile,json
W=Path(__file__).resolve().parent
waiting=Path('/proc/137453')
if waiting.exists():
 assert waiting.stat().st_uid==os.getuid() and b'queue_timeonly_ablation.py' in (waiting/'cmdline').read_bytes()
 assert not (waiting/'task/137453/children').read_text().strip(),'Ablation already started; do not stop an active experiment'
 os.kill(137453,signal.SIGTERM)
s=(W/'run_progress_guarded.py').read_text().split('\ntry:\n check()')[0]
s=s.replace("OUT=W/'progress_act_20261125'","OUT=W/'receding_act_20261125'")
needle="evaluation=launch([PY,str(W/'evaluate_g2_policy.py')"
assert needle in s
s=s.replace(needle,"extra+=['--execute','1','--assets','bin_lidded','bin_pedal','basket_rectangular','box_white']\n "+needle)
s+='''
try:
 check();s=snapshot()
 if s['available_gib']<64 or any(g[1]<20000 for g in s['gpus'] if g[0] in (0,1)):raise RuntimeError('Insufficient startup headroom')
 run_eval('progress_act',W/'runs/act_progress_v1/step_008000.pt','act_progress_v1_execute1','pilot4_one_each',True)
 (OUT/'finished.json').write_text(json.dumps(dict(status='evaluation_complete',time=time.strftime('%FT%T'))))
except Exception as ex:
 (OUT/'stopped.json').write_text(json.dumps(dict(error=repr(ex),time=time.strftime('%FT%T'))));raise
finally:
 for child in reversed(children):stop(child)
'''
(W/'run_receding_guarded.py').write_text(s);py_compile.compile(str(W/'run_receding_guarded.py'),doraise=True)
print('RECEDING_EVAL_PREPARED; old waiting ablation cancelled')
