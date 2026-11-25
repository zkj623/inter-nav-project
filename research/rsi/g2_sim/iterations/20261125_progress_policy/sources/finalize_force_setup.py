from pathlib import Path
import py_compile
W=Path(__file__).resolve().parent
p=W/'evaluate_force_policy.py';s=p.read_text(encoding='utf-8-sig')
old="  result=json.loads((T/'result.json').read_text());v=json.loads((T/'verification.json').read_text()) if (T/'verification.json').exists() else {}"
assert s.count(old)==1
new=old+"\n  with (out/(asset+'_force_input_audit.log')).open('w') as log:\n   force_audit_rc=subprocess.run([sys.executable,str(W/'audit_force_episode.py'),'--trial',str(T)],stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode\n  if force_audit_rc:v['verified_success']=False;v['failure']='FORCE_INPUT_CAUSALITY_AUDIT_FAILED'"
p.write_text(s.replace(old,new));py_compile.compile(str(p),doraise=True)
print('FORCE_INPUT_AUDIT_REQUIRED')
