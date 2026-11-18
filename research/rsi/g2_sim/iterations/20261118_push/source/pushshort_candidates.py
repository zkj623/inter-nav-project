from pathlib import Path
import json
def candidates(aid):
 cs=json.loads((Path(__file__).resolve().parent/f'pushsafe_geometry_{aid}.json').read_text())['candidates']
 cs=[c for c in cs if c['yaw']==90 and abs(c['contact_height']-.22)<1e-6]
 return cs,dict(cached_22cm_candidates=len(cs))
