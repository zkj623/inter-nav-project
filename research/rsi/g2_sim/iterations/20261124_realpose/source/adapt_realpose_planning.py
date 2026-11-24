from pathlib import Path
import json
R=Path('/data20t/embodied/zkj/inter-nav-results/g2-realpose-20261124')
for p in (R/'realpose_plans').glob('*_planning.json'):
 d=json.loads(p.read_text());print(d['asset'],d['success'],d.get('reason',''))
p=R/'plan_realpose.py';s=p.read_text();s=s.replace("pa.add_argument('--asset',required=True);args=pa.parse_args()","pa.add_argument('--asset',required=True);pa.add_argument('--dx',type=float,default=0.);args=pa.parse_args()")
s=s.replace("target=targets[f'arm_{side}_end_link']@tcp", "target=targets[f'arm_{side}_end_link']@tcp;target[0,3]+=args.dx")
s=s.replace("    for key in ['q_pre'", "    p['object_position']=(np.array(old['object_position'])+[args.dx,0,0]).tolist()\n    p['realpose_object_shift_m']=[args.dx,0,0]\n    report['object_shift_x_m']=args.dx\n    for key in ['q_pre'")
p.write_text(s)
