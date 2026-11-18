from pushnear_candidates import candidates as original
def candidates(aid):
 cs,diag=original(aid)
 cs.sort(key=lambda c:abs(c['contact_height']-.27)+(.02 if c['wrist_roll_variant']!='original' else 0))
 return cs,diag
