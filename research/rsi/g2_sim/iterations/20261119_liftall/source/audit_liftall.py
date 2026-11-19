from pathlib import Path
import json,sys,subprocess,os
R=Path(__file__).resolve().parent;pid=int(sys.argv[1]);folder='liftall_validated_plans';out=R/folder;out.mkdir(exist_ok=True)
rows=json.loads((R/f'marginpose_search_{pid}.json').read_text());assets=[]
for row in rows:
 if not row['passed']:continue
 aid=row['asset'];assets.append(aid);(out/(aid+'.json')).write_text(json.dumps(dict(asset=aid,plans=[row['plan']]),indent=2))
for kind,src in [('visible','strategy15_support_validate_plans.py'),('native','strategy15_support_audit_native_self.py'),('object','strategy15_support_audit_object_clearance.py')]:
 text=(R/src).read_text().replace('strategy15_support_validated_plans',folder).replace('strategy15_support_plans',folder)
 text=text.replace("aid=='stool_rectangular'","aid in ['stool_rectangular', 'stool_wood_four_leg', 'table_side', 'basket_rectangular']")
 script=R/f'liftall_{kind}_audit.py';script.write_text(text)
 for aid in assets:
  subprocess.run([sys.executable,str(script),aid],check=True,cwd=R)
summary={aid:len(json.loads((out/(aid+'.json')).read_text())['plans']) for aid in assets}
(R/'liftall_audit_summary.json').write_text(json.dumps(summary,indent=2));print('AUDIT_DONE',pid,summary,flush=True)
