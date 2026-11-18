from pathlib import Path
import json
from chaircorners_wide_candidates import candidates as wide
from chaircorners_narrow_candidates import candidates as narrow
def candidates(aid):
 out=[];counts={}
 for label,q,fn in [('wide',-.785,wide),('narrow',-.2,narrow)]:
  cs,stats=fn(aid);counts[label]=stats
  for c in cs:c['opening']=q
  out.extend(cs)
 Path(__file__).with_name('corners_geometry_'+aid+'.json').write_text(json.dumps(dict(candidates=out,counts=counts),indent=2))
 return out,counts
