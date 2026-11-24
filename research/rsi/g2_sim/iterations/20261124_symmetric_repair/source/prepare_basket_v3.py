from pathlib import Path
import json
R=Path(__file__).resolve().parent;O=R/'repair_v3';O.mkdir(exist_ok=True)
d=json.loads((R/'realpose_plans/basket_rectangular.json').read_text());d['plans'][0]['repair_revision']='v3 original validated rim depth and positions; lower gripper stiffness5 damping0.5'
(O/'basket_rectangular.json').write_text(json.dumps(d,indent=2))
