from pathlib import Path
import os,signal,subprocess,json,time
W=Path(__file__).resolve().parent;pid=154384;P=Path('/proc')/str(pid)
assert P.stat().st_uid==os.getuid()
argv=(P/'cmdline').read_bytes().split(b'\0')
assert str(W/'queue_receding_eval.py').encode() in argv
assert not (W/'timeonly_act_20261125').exists(), 'Time-only work already began; inspect before changing queue'
assert json.loads((W/'force_model_test.json').read_text())['contact_gradient_nonzero']
# Stop only the waiting scheduler PID. The active guarded evaluation is its own
# process group and continues with its existing resource monitor and lock.
os.kill(pid,signal.SIGTERM)
new=subprocess.Popen(['/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python',str(W/'queue_force_correction.py')],cwd=W,stdout=(W/'force_queue.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
(W/'force_queue_transition.json').write_text(json.dumps(dict(time=time.strftime('%FT%T'),stopped_scheduler_pid=pid,active_guard_preserved=157579,new_scheduler_pid=new.pid,reason='Faster feedback did not fix lidded-bin instability; prioritize causal measured contact inputs over removing visual/proprioceptive feedback'),indent=2))
print('FORCE_QUEUE_PID',new.pid)
