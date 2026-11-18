from pathlib import Path
import json
from chairasym_leftwide_candidates import candidates as leftwide
from chairasym_rightwide_candidates import candidates as rightwide
def candidates(aid):
 out=[];counts={}
 for name,ql,qr,fn in [('leftwide',-.785,0.,leftwide),('rightwide',0.,-.785,rightwide)]:
  cs,stats=fn(aid);counts[name]=stats
  for c in cs:c['opening_left']=ql;c['opening_right']=qr;c['opening']=ql
  out.extend(cs)
 Path(__file__).with_name('asym_geometry_'+aid+'.json').write_text(json.dumps(dict(candidates=out,counts=counts),indent=2))
 return out,counts
