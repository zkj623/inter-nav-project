from supportpalmdown_wide_candidates import candidates as wide
from supportpalmdown_zero_candidates import candidates as zero
from supportpalmdown_narrow_candidates import candidates as narrow
def candidates(aid):
 out=[];counts={}
 for label,q,fn in [('wide',-.785,wide),('closed',0.,zero),('narrow',-.2,narrow)]:
  cs,stats=fn(aid);counts[label]=stats
  for c in cs:c['opening']=q
  out.extend(cs)
 return out,counts
