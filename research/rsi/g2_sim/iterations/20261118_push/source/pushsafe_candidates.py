from pushsafe_geometry import candidates as original
def candidates(aid):
 cs,diag=original(aid);cs=[c for c in cs if c['yaw']==90]
 cs.sort(key=lambda c:abs(c['contact_height']-.22))
 return cs,diag
