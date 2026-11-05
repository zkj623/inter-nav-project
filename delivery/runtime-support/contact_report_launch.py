"""Observe PhysX contact pairs during a diagnostic; no control replacement."""
import collections,hashlib,json,runpy,shutil,sys
from pathlib import Path
import numpy as np
import grutopia_extension
from grutopia.core import gym_env
from grutopia_extension.interactive_navigation import go2_navigation_runner

entry=sys.argv.pop(1)
flag='--output' if '--output' in sys.argv else '--record-dir'
output=Path(sys.argv[sys.argv.index(flag)+1]);output.mkdir(parents=True,exist_ok=True)
source=output/'contact_report_launch.py';shutil.copy2(__file__,source)
OriginalEnv=gym_env.Env
configured_bodies=[]
original_import_extensions=grutopia_extension.import_extensions

def import_with_contact_reporting(*args,**kwargs):
    result=original_import_extensions(*args,**kwargs)
    from grutopia_extension.robots.go2 import Go2
    if not getattr(Go2,'_contact_probe_patched',False):
        original_init=Go2.__init__
        def init(self,*a,**kw):
            original_init(self,*a,**kw)
            from pxr import Usd,UsdPhysics,PhysxSchema
            from omni.isaac.core.utils.stage import get_current_stage
            root=get_current_stage().GetPrimAtPath(getattr(self,'_go2_asset_root_path',self.prim_path))
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
                    configured_bodies.append(str(prim.GetPath()))
        Go2.__init__=init;Go2._contact_probe_patched=True
    return result

class ContactEnv(OriginalEnv):
    def reset(self,*args,**kwargs):
        result=super().reset(*args,**kwargs)
        from pxr import Usd,UsdPhysics,PhysxSchema,PhysicsSchemaTools
        from omni.isaac.core.utils.stage import get_current_stage
        from omni.physx import get_physx_simulation_interface
        stage=get_current_stage()
        task=next(iter(self.runner.current_tasks.values()));robot=next(iter(task.robots.values()))
        self._contact_ticks=0;self._contact_command=None;self._contact_pose=None;self._contact_orientation=None
        self._contact_stage=stage;self._contact_task=None
        self._contact_counts=collections.Counter();self._contact_errors=[];self._contact_written=0
        self._contact_file=(output/'contacts.jsonl').open('w',buffering=1)
        self._control_trace=(output/'control_trace.jsonl').open('w',buffering=1)
        def path(value):
            try:return str(PhysicsSchemaTools.intToSdfPath(value))
            except Exception:return str(value)
        def report(headers,data):
            try:
                for header in headers:
                    a=path(header.actor0);b=path(header.actor1)
                    pair=str(self._contact_task)+' :: '+a+' | '+b;self._contact_counts[pair]+=1
                    floor='floor' in (a+' '+b).lower() or 'ground' in (a+' '+b).lower()
                    points=[]
                    for i in range(header.contact_data_offset,header.contact_data_offset+header.num_contact_data):
                        item=data[i];point={}
                        for k in ['position','normal','impulse']:
                            value=getattr(item,k)
                            point[k]=float(value) if isinstance(value,(float,int)) else [float(value[j]) for j in range(3)]
                        point['separation']=float(getattr(item,'separation',0.))
                        points.append(point)
                    unusual_floor=any(abs(p['normal'][2])<.95 or p['position'][2]>.05 for p in points)
                    if floor and self._contact_counts[pair]>20 and not unusual_floor:continue
                    row={'env_tick':self._contact_ticks,'task':self._contact_task,'type':str(header.type),'actor0':a,'actor1':b,
                        'collider0':path(getattr(header,'collider0',0)),'collider1':path(getattr(header,'collider1',0)),
                        'command':self._contact_command,'previous_robot_position':self._contact_pose,
                        'previous_robot_orientation':self._contact_orientation,'points':points}
                    self._contact_file.write(json.dumps(row)+'\n');self._contact_written+=1
            except Exception as e:
                if len(self._contact_errors)<10:self._contact_errors.append(repr(e))
        self._contact_subscription=get_physx_simulation_interface().subscribe_contact_report_events(report)
        (output/'contact_setup.json').write_text(json.dumps({'instrumentation_only':True,'rigid_bodies':configured_bodies,
            'wrapper_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'entry':entry,
            'repo':str(Path.cwd()),'floor_contacts_sampled':True},indent=2))
        return result

    def step(self,action):
        self._contact_command=action.get('move_by_speed')
        marker=self._contact_stage.GetPrimAtPath('/World/TaskMarkers/Goal')
        self._contact_task=marker.GetCustomDataByKey('task_id') if marker else None
        result=super().step(action)
        self._contact_pose=np.asarray(result[0]['position']).tolist()
        self._contact_orientation=np.asarray(result[0]['orientation']).tolist()
        self._control_trace.write(json.dumps({'env_tick':self._contact_ticks,'task':self._contact_task,
            'command':self._contact_command,'position':self._contact_pose,
            'orientation':self._contact_orientation})+'\n')
        self._contact_ticks+=1
        return result

    def close(self):
        if hasattr(self,'_contact_subscription'):
            self._contact_subscription=None
            (output/'contact_summary.json').write_text(json.dumps({'event_counts':dict(self._contact_counts),
                'written':self._contact_written,'errors':self._contact_errors,'env_ticks':self._contact_ticks},indent=2))
            self._contact_file.close()
            self._control_trace.close()
        return super().close()

gym_env.Env=ContactEnv
go2_navigation_runner.Env=ContactEnv
grutopia_extension.import_extensions=import_with_contact_reporting
go2_navigation_runner.import_extensions=import_with_contact_reporting
sys.argv[0]=entry
runpy.run_path(entry,run_name='__main__')
