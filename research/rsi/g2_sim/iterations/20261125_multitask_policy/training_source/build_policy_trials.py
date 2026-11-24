from pathlib import Path
import json,ast
R=Path('/data20t/embodied/zkj/inter-nav-results/g2-act-threecam-20261124')
assets=list(json.loads((R/'standby_preparation_config.json').read_text())['execution'])
for asset in assets:
 s=(R/f'realpose_{asset}_trial.py').read_text()
 s=s.replace('args=p.parse_args()',"p.add_argument('--policy-url',default='http://127.0.0.1:19024');p.add_argument('--checkpoint',required=True);p.add_argument('--run',required=True);p.add_argument('--duration',type=float,default=80.);p.add_argument('--execute',type=int,default=10);args=p.parse_args()")
 s=s.replace("ROOT/'realpose_trials'/args.plan_dir","ROOT/'policy_trials'/args.run/args.plan_dir")
 s=s.replace('from threecam_recorder import Recorder','from policy_threecam_recorder import Recorder')
 line=next(line for line in s.splitlines() if line.startswith('recorder=Recorder('));s=s.replace(line,line+"\nrecorder.configure_policy(args.policy_url,args.asset,args.checkpoint,args.execute)")
 start=s.index('    settled_z=float(obj.get_world_pose()[0][2])');end=s.index("    result['collection_timing_profile']",start)
 s=s[:start]+'''    settled_z=float(obj.get_world_pose()[0][2])
    result['cartesian_feedback']='DISABLED: learned policy only'
    result['feedback_revision']='DISABLED: learned policy only'
    for control_step in range(round(args.duration*20)):
        phase='common_approach' if control_step==0 else ('hold' if control_step>=round((args.duration-3.1)*20) else 'policy')
        for _ in range(6):tick(plan['q_start'],phase)
'''+s[end:]
 s=s.replace('G2 VARIED-PLACEMENT COLLECTION','G2 LEARNED POLICY')
 ast.parse(s);(R/f'policy_{asset}_trial.py').write_text(s)
print('POLICY_HARNESSES_READY',len(assets))
