from pathlib import Path
import shutil,json,hashlib,subprocess,datetime
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';P=W.parent/'g2-publish-20261118';O=P/'research/rsi/g2_sim/iterations/20261125_multitask_policy';O.mkdir(parents=True,exist_ok=True)
files={}
for kind,paths in [('training_source',list(W.glob('*.py'))),('simulation_source',[p for p in R.glob('*.py') if p.name.startswith(('policy_','threecam_recorder','plan_wide','run_wide','wide_posture','verify_threecam'))])]:
 d=O/kind;d.mkdir(exist_ok=True)
 for p in paths:
  shutil.copy2(p,d/p.name);files[str((d/p.name).relative_to(O))]=hashlib.sha256(p.read_bytes()).hexdigest()
for repo in ['act','openpi']:
 d=W/'vendor'/repo
 (O/(repo+'_upstream.json')).write_text(json.dumps(dict(remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=d,text=True).strip(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d,text=True).strip()),indent=2))
 (O/(repo+'_local.patch')).write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=d))
dest=O/'results';dest.mkdir(exist_ok=True)
for p in W.glob('*.json'):
 if p.stat().st_size<2_000_000:shutil.copy2(p,dest/p.name)
for folder in ['datasets','evaluations','runs','pi05_sources']:
 for p in (W/folder).rglob('*'):
  if p.is_file() and p.suffix in ['.json','.jsonl','.py'] and p.stat().st_size<10_000_000:
   out=dest/folder/p.relative_to(W/folder);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
(O/'source_manifest.json').write_text(json.dumps(dict(snapshot_time=datetime.datetime.now().isoformat(),sha256=files,remote_workspace=str(W),simulation_root=str(R)),indent=2))
(O/'.gitattributes').write_text('training_source/*.py -text whitespace=cr-at-eol,-blank-at-eof\nsimulation_source/*.py -text whitespace=cr-at-eol,-blank-at-eof\nresults/**/*.py -text whitespace=cr-at-eol,-blank-at-eof\n*.patch -text whitespace=-blank-at-eol,-blank-at-eof\n')
shutil.copy2(W/'README_multitask_policy.md',O/'README.md')
shutil.copy2(W/'README_multitask_policy.md',O/'training_source/README_multitask_policy.md')
print(O)

