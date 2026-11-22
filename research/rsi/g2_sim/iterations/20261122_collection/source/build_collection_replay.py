from pathlib import Path
import ast,hashlib
R=Path(__file__).resolve().parent;v=R/'source_versions';v.mkdir(exist_ok=True)
for p in R.glob('collect_*_trial.py'):
 s=p.read_text();(v/(hashlib.sha256(p.read_bytes()).hexdigest()+'.py')).write_bytes(p.read_bytes());s=s.replace("ROOT/'collection_trials'","ROOT/'collection_replay_trials'")
 s=s.replace('args=p.parse_args()',"p.add_argument('--replay-episode',required=True);args=p.parse_args()")
 start=s.index("    if plan and 'q_start' in plan and not args.pose_only:");end=s.index("    result['collection_initialization']=",start)
 replacement="""    import h5py
    with h5py.File(args.replay_episode,'r') as dataset:
        replay_actions=dataset['action'][:];replay_phases=dataset['phase'].asstr()[:]
    result['replay_input_sha256']=hashlib.sha256(Path(args.replay_episode).read_bytes()).hexdigest()
    result['replay_scope']='open-loop recorded actions only; no expert retarget, object tracking or force-balancing calls'
    for action,phase in zip(replay_actions,replay_phases):
        target={**plan['q_start'],**dict(zip(recorder.names,map(float,action)))}
        for _ in range(6):tick(target,str(phase))
"""
 s=s[:start]+replacement+s[end:];ast.parse(s);out=R/p.name.replace('collect_','replay_',1);out.write_text(s);(v/(hashlib.sha256(out.read_bytes()).hexdigest()+'.py')).write_bytes(out.read_bytes())
print('REPLAY_SOURCES_BUILT')
