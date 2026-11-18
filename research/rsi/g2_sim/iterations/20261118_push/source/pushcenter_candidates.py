from pushnear_candidates import candidates as original
def candidates(aid):
 cs,diag=original(aid)
 cs=[c for c in cs if c['yaw'] in (180,0) and all(abs(p[1])<.003 for p in c['contact_points'].values())]
 cs.sort(key=lambda c:abs(c['contact_height']-.28)+(.02 if c['wrist_roll_variant']!='original' else 0)+(.01 if c['yaw']==0 else 0))
 diag['central_symmetry_candidates']=len(cs)
 return cs,diag
