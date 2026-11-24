"""Audit an RSI standby initialization and free-base holding recording."""
import hashlib,json,math
from pathlib import Path
import argparse

def audit(folder):
    folder=Path(folder)
    load=lambda name:json.loads((folder/name).read_text())
    r=load('result.json');q=load('pose.json');config=load('standby_preparation_config.json');source=load('pose-source.json');footprint=load('g2_collision_footprint.json')
    non_support=0
    for event in load('contacts.json'):
        for point in event['points']:
            impulse=math.sqrt(sum(v*v for v in point['impulse']))
            if impulse>.001 and not (abs(point['normal'][2])>.7 and point['position'][2]<.08):non_support+=1
    checks={'exact_requested_joint_targets':q==config['standby_joint_positions_rad'],
            'configuration_sha256':hashlib.sha256((folder/'standby_preparation_config.json').read_bytes()).hexdigest()==source['config_sha256'],
            'posture_specific_footprint':footprint['pose_source']['config_sha256']==source['config_sha256'],
            'hold_completed':r.get('success') is True and r.get('reason')=='standby_hold_check_complete' and r.get('simulation_seconds',0)>=5,
            'held_joint_error':r.get('standby_joint_max_error_rad',math.inf)<.05,
            'stable_body':r.get('maximum_tilt_deg',math.inf)<5,
            'stopped':r.get('final_speed_m_s',math.inf)<.03 and r.get('final_angular_speed_rad_s',math.inf)<.05,
            'no_force_bearing_self_contacts':not load('self-contacts.json'),
            'no_force_bearing_non_support_contacts':non_support==0}
    report={'passed':all(checks.values()),'checks':checks,'configured_joints':len(q),'maximum_joint_error_rad':r.get('standby_joint_max_error_rad'),'non_support_force_points':non_support,'self_contact_events':len(load('self-contacts.json')),'scope':'Exact default posture initialization and five-second free-base holding check; not a completed navigation episode.'}
    (folder/'standby-audit.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');report=audit(p.parse_args().folder);print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
