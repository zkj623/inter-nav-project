"""Observe exact production policy inputs and outputs without changing controls."""
from pathlib import Path
import atexit,hashlib,json,runpy,sys
import numpy as np
import grutopia_extension
from grutopia_extension.interactive_navigation import go2_navigation_runner

entry=sys.argv.pop(1)
output=Path(sys.argv[sys.argv.index('--output')+1])
output.mkdir(parents=True,exist_ok=True)
source=Path(__file__)
(output/'policy_state_capture.py').write_bytes(source.read_bytes())
original_import=grutopia_extension.import_extensions
handles=[]
def close():
    for stream in handles:
        if not stream.closed:stream.close()
atexit.register(close)

def import_and_observe(*args,**kwargs):
    result=original_import(*args,**kwargs)
    from grutopia_extension.controllers import isaac_go2_ctrl as module
    cls=module.Go2RSLControl
    if getattr(cls,'_v193_state_capture',False):return result
    cls._v193_state_capture=True
    stream=(output/'policy_state_trace.jsonl').open('w',buffering=1)
    motor_stream=(output/'motor_state_trace.jsonl').open('w',buffering=1)
    handles.extend([stream,motor_stream])
    build=module.build_go2_observation
    infer=cls._infer_policy
    step=cls.step
    captured={}
    def observe_build(*a,**kw):
        observation=build(*a,**kw)
        captured.clear()
        names=['body_linear_velocity','body_angular_velocity','projected_gravity','command',
               'joint_positions','default_joint_positions','joint_velocities','previous_action']
        for key,value in zip(names,a):captured[key]=np.asarray(value).tolist()
        captured['observation']=np.asarray(observation).tolist()
        return observation
    def observe_infer(self):
        result=infer(self)
        position,orientation=self.articulation.get_world_pose()
        row={'time_s':float(self._simulation_time()),'inference':self.policy_inferences,
             **captured,'position':np.asarray(position).tolist(),'orientation':np.asarray(orientation).tolist(),
             'raw_action':self.raw_policy_action.tolist(),'action':self.last_action.tolist(),
             'motor_target':self.applied_joint_positions.tolist(),'policy_command':self.policy_command.tolist()}
        stream.write(json.dumps(row)+'\n')
        if self.policy_inferences%100==0:
            p=output/'policy_capture_live.tmp';p.write_text(json.dumps({'time_s':row['time_s'],'inferences':self.policy_inferences,
                'position':row['position'],'policy_command':row['policy_command'],'capture_only':True}));p.replace(output/'policy_capture_live.json')
        return result
    def observe_step(self):
        result=step(self)
        now=float(self._simulation_time())
        if now>=119.:
            motor_stream.write(json.dumps({'time_s':now,'inference':self.policy_inferences,
                'joint_positions':self.joint_subset.get_joint_positions().tolist(),
                'joint_velocities':self.joint_subset.get_joint_velocities().tolist(),
                'motor_target':self.applied_joint_positions.tolist(),'applied_efforts':self.applied_joint_efforts.tolist()})+'\n')
        return result
    module.build_go2_observation=observe_build
    cls._infer_policy=observe_infer
    cls.step=observe_step
    (output/'policy_capture_contract.json').write_text(json.dumps({'instrumentation_only':True,
        'original_task_accepted':False,'wrapper_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'observation_source':'Exact original build_go2_observation arguments and returned observation',
        'action_source':'Original Go2RSLControl._infer_policy return, no extra actor inference',
        'motor_source':'Original Go2RSLControl.step return; read-only joint query afterward',
        'writes_to_controls':False,'physical_resets_added':False,'motor_capture_from_s':119.},indent=2)+'\n')
    return result

grutopia_extension.import_extensions=import_and_observe
go2_navigation_runner.import_extensions=import_and_observe
sys.argv[0]=entry
runpy.run_path(entry,run_name='__main__')
