from pathlib import Path
import json
R=Path('/data20t/embodied/zkj/inter-nav-results/g2-torquelow-20261121');v=R/'vendor/genie_sim_drive_audit/source/geniesim_ros';print((v/'src/ros_ws/src/genie_sim_bringup/config/scene_flat_g2_op.yaml').read_text()[:2500]);p=v/'src/ros_ws/src/genie_sim_engine_core/genie_sim_engine_core/common/params.py';ls=p.read_text().splitlines();print('\n'.join(f'{i+1}: {ls[i]}' for i in range(535,553)))
for f in (v/'src/ros_ws/src').rglob('physics*yam*'):
 s=f.read_text()
 if 'body:' in s:print('PARAM_FILE',f,[(i,l) for i,l in enumerate(s.splitlines(),1) if 'body:' in l or 'max_effort:' in l][:15])
