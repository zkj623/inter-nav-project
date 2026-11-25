from pathlib import Path
import py_compile
W=Path(__file__).resolve().parent
anchor='children=[]; previous=None; strikes=0'
replacement=anchor+"\nos.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])\npriority=os.getpriority(os.PRIO_PROCESS,0)\nif priority<10:os.nice(10-priority)"
for name in ['resume_guarded_eval.py','run_progress_guarded.py','run_timeonly_guarded.py']:
 p=W/name;s=p.read_text()
 if 'os.sched_setaffinity' not in s:
  assert anchor in s;s=s.replace(anchor,replacement);tmp=p.with_suffix('.updated.py');tmp.write_text(s);py_compile.compile(str(tmp),doraise=True);tmp.replace(p)
print('CPU_LIMITS_PERSISTED')
