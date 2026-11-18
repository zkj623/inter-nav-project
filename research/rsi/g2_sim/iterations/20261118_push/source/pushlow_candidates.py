from pushnear_candidates import candidates as original
def candidates(aid):
 cs,diag=original(aid)
 cs=[c for c in cs if c['contact_height']<=.20 and c['yaw'] in (90,270)]
 cs.sort(key=lambda c:abs(c['contact_height']-.14)+(.02 if c['wrist_roll_variant']!='original' else 0)+(.01 if c['yaw']==270 else 0))
 diag['low_push_candidates']=len(cs)
 return cs,diag
