"""Check a newly reproduced task; preserve the historical audits as separate evidence."""
from pathlib import Path
import argparse,json,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tools.audit_actual_passage import final_stop_evidence
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('task');a=p.parse_args();root=a.output;f=root/a.task
r=json.loads((f/'result.json').read_text());c=[json.loads(l) for l in (f/'control_trace.jsonl').read_text().splitlines()];e=[json.loads(l) for l in (f/'events.jsonl').read_text().splitlines()]
stop=final_stop_evidence(c,r['goal']);guard=json.loads((root/'online_input_audit.json').read_text());startup=next(x for x in e if x['event']=='startup');ends=[x for x in e if x['event']=='skill_end']
checks={'goal_reached':r['success'] and r['reason']=='goal_reached','goal_error_le_05m':r['goal_error_m']<=.5,'measured_stop':stop['valid'],'tilt_below_30':r['maximum_tilt_degrees']<30,'startup_below_15':startup['passed'] and startup['tilt_degrees']<15,'healthy':r['rollout_health']['healthy'],'online_input_guard':guard['guard_armed'] and not guard['prior_read_attempts'],'interaction_completed':int(a.task[1:])<=5 or (r['push_count']>=1 and bool(ends) and all(x['status']=='succeeded' and x.get('actual_push_distance_m',0)>.25 for x in ends))}
report={'passed':all(checks.values()),'checks':checks,'goal_error_m':r['goal_error_m'],'scope':'New-machine task completion and runtime skill checks; historical full physics/contact/passage audits remain in delivery/evidence.'};(root/'reproduction-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));sys.exit(0 if report['passed'] else 1)
