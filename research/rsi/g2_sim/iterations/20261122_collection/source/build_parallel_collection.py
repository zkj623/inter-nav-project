from pathlib import Path
import ast,hashlib
R=Path(__file__).resolve().parent;v=R/'source_versions';v.mkdir(exist_ok=True)
for p in R.glob('collect_*_trial.py'):
 if p.stem.endswith('_parallel_trial'):continue
 s=p.read_text();needle='try:\n    OUT.mkdir(parents=True,exist_ok=False)';assert needle in s
 s=s.replace(needle,"try:\n    import os\n    if os.environ.get('G2_RESERVED_TRIAL')==str(OUT.resolve()):\n        assert json.loads((OUT/'reservation.json').read_text())['owner_pid']==os.getppid()\n    else:OUT.mkdir(parents=True,exist_ok=False)")
 ast.parse(s);out=R/p.name.replace('_trial.py','_parallel_trial.py');out.write_text(s);(v/(hashlib.sha256(out.read_bytes()).hexdigest()+'.py')).write_bytes(out.read_bytes())
print('PARALLEL_COLLECTION_SOURCES_READY')
