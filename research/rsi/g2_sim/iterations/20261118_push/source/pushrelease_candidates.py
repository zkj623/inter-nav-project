from pushrelease_geometry import candidates as original
def candidates(aid):
 cs,diag=original(aid);cs=[c for c in cs if c['yaw']==90];return cs,diag
