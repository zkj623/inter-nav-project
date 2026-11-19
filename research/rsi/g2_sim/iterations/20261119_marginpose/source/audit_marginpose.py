from pathlib import Path
import json,sys,subprocess,os
R=Path(__file__).resolve().parent;pid=int(sys.argv[1]);folder=f'marginpose_selected{pid}_validated_plans';out=R/folder;out.mkdir(exist_ok=True)
rows=json.loads((R/f'marginpose_search_{pid}.json').read_text());assets=[]
for row in rows:
 if not row['passed']:continue
 aid=row['asset'];assets.append(aid);(out/(aid+'.json')).write_text(json.dumps(dict(asset=aid,plans=[row['plan']]),indent=2))
for kind,src in [('visible','strategy15_support_validate_plans.py'),('native','strategy15_support_audit_native_self.py'),('object','strategy15_support_audit_object_clearance.py')]:
 text=(R/src).read_text().replace('strategy15_support_validated_plans',folder).replace('strategy15_support_plans',folder)
 script=R/f'marginpose_selected{pid}_{kind}_audit.py';script.write_text(text)
 for aid in assets:
  subprocess.run([sys.executable,str(script),aid],check=True,cwd=R)
summary={aid:len(json.loads((out/(aid+'.json')).read_text())['plans']) for aid in assets}
(R/f'marginpose_selected{pid}_audit_summary.json').write_text(json.dumps(summary,indent=2));print('AUDIT_DONE',pid,summary,flush=True)
