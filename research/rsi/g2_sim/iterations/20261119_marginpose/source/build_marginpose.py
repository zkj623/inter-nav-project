from pathlib import Path
import shutil,json,ast
R=Path(__file__).resolve().parent
A=R.parent/'g2-publish-20261118/research/rsi/g2_sim/iterations/20261119_fixedpose'
for directory in ['source','fixtures']:
 for p in (A/directory).rglob('*'):
  if p.is_file():
   dest=R/p.relative_to(A/directory);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
s=(R/'search_fixedpose.py').read_text()
start=s.index('profiles=');end=s.index('\npid=',start)
s=s[:start]+'''profiles=[[-50,140,-75,0,0],[-45,140,-80,0,0],[-40,140,-85,0,0],[-35,140,-90,0,0],[-30,140,-95,0,0],[-45,135,-75,0,0]]'''+s[end:]
s=s.replace("fixedpose_search_{pid}","marginpose_search_{pid}")
s=s.replace("for dx in [0.,-.04,.04,-.08,.08]:","for dx in [0.,-.04,.04,-.08,.08,-.12,.12]:")
s=s.replace("for row in rows:","""for n,v in body.items():
 j=k.by_name[n];assert min(v-j['lower'],j['upper']-v)>=np.radians(10),n
for row in rows:""")
ast.parse(s);(R/'search_marginpose.py').write_text(s)
s=(R/'audit_selected_fixedpose.py').read_text().replace('fixedpose_selected','marginpose_selected').replace('fixedpose_search','marginpose_search')
(R/'audit_marginpose.py').write_text(s)
print('Prepared margin-constrained search; all torso joints require >=10 deg margin')
