from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent
s=(R/'verify_liftpair.py').read_text().replace("R.glob('liftpair*_trials/**/result.json')","R.glob('torquelow_reference_trials/**/result.json')").replace('liftpair_verified_results.json','torquelow_reference_verified_results.json');s=s.replace("for s in samples for n,j in k.by_name.items() if 'body_joint' in n","for s in samples if s['phase'] not in ('default_settle','prepare_home','initial_settle') for n,j in k.by_name.items() if 'body_joint' in n");exec(compile(s,'finite1200_grasp_review','exec'))
from gravity_compensation import GravityCompensation
g=GravityCompensation();k=g.e.k;rows=json.loads((R/'torquelow_reference_verified_results.json').read_text())
for row in rows:
 f=R/row['result'];d=json.loads(f.read_text());ss=json.loads(f.with_name('samples.json').read_text());row['source_matches']=hashlib.sha256((R/'torquelow_reference_trial.py').read_bytes()).hexdigest()==d['source_sha256'];row['waist_unlocked']=d.get('waist_hold_is_unlocked');row['waist_effort_limits_nm']=d.get('waist_effort_limits_nm');row['max_self_contact_impulse_ns']=max((v['impulse'] for v in json.loads(f.with_name('self_contacts.json').read_text())),default=0.);row['empty_model_waist_gravity_peak_nm']=max(abs(t) for s in ss for n,t in g.torques(s['joint_positions']).items() if 'body_joint' in n);row['verified_success']=bool(row['verified_success'] and row['source_matches'] and row['waist_unlocked'] and all(v==1200 for v in row['waist_effort_limits_nm'].values()) and row['max_self_contact_impulse_ns']<=.001)
 print('FINITE1200_REVIEW',row['verified_success'],row['empty_model_waist_gravity_peak_nm'],flush=True)
(R/'torquelow_reference_verified_results.json').write_text(json.dumps(rows,indent=2))
