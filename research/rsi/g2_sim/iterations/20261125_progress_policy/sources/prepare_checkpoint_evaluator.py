from pathlib import Path
import py_compile
W=Path(__file__).resolve().parent
s=(W/'run_force_guarded.py').read_text();s=s[:s.index('\ntry:\n check()')]
insert='''
import argparse,re
parser=argparse.ArgumentParser(description='Evaluate an existing contact-input ACT checkpoint without retraining; guarded and serial')
parser.add_argument('--run',required=True);parser.add_argument('--checkpoint',default=str(W/'runs/act_force_v1/step_008000.pt'));parser.add_argument('--assets',nargs='+',choices=['box_white','bin_lidded','bin_pedal','basket_rectangular']);parser.add_argument('--execute',type=int,default=2)
a=parser.parse_args();assert re.fullmatch(r'[a-zA-Z0-9_-]+',a.run) and 1<=a.execute<=50
assert not (W/'evaluations'/a.run).exists(),'Use a new run name to preserve previous results'
'''
s=s.replace('W=Path(__file__).resolve().parent','W=Path(__file__).resolve().parent'+insert)
s=s.replace("OUT=W/'force_act_20261125'; OUT.mkdir(exist_ok=True)","OUT=W/'checkpoint_reviews'/a.run; OUT.mkdir(parents=True,exist_ok=False)")
s=s.replace("'--execute','2']+extra","'--execute',str(a.execute)]+extra+(['--assets',*a.assets] if a.assets else [])")
s+='''
def interrupted(signum,frame):raise RuntimeError('REQUESTED_STOP_SIGNAL_'+str(signum))
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
 check();s=snapshot()
 if s['available_gib']<64 or any(g[1]<20000 for g in s['gpus'] if g[0] in (0,1)):raise RuntimeError('Insufficient startup headroom')
 run_eval('force_act',Path(a.checkpoint),a.run,'pilot4_force_causal',True)
 (OUT/'finished.json').write_text(json.dumps(dict(time=time.strftime('%FT%T'),status='evaluations_complete')))
except Exception as ex:
 (OUT/'stopped.json').write_text(json.dumps(dict(time=time.strftime('%FT%T'),error=repr(ex))));raise
finally:
 for child in reversed(children):stop(child)
'''
(W/'evaluate_checkpoint_guarded.py').write_text(s);py_compile.compile(str(W/'evaluate_checkpoint_guarded.py'),doraise=True)
print('CHECKPOINT_REVIEW_ENTRY_READY_NO_JOB_STARTED')
