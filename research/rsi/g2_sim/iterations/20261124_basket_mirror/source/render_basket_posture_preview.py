from pathlib import Path
import json,sys,numpy as np
R=Path(__file__).resolve().parent;OLD=R.parent/'g2-act-collect-20261122';O=R/'posture_preview';O.mkdir(exist_ok=True)
sys.argv.extend(['--/plugins/carb.tasking.plugin/threadCount=12','--/plugins/omni.tbb.globalcontrol/maxThreadCount=12'])
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':2,'multi_gpu':False})
try:
    import cv2,omni.usd,omni.timeline,omni.replicator.core as rep
    from scipy.spatial.transform import Rotation
    from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
    from omni.isaac.core import World
    from omni.isaac.sensor import Camera
    from omni.isaac.core.utils.semantics import add_update_semantics
    from g2_kinematics import Kinematics,transform
    from home_floor_material import add_wood_floor
    cfg=json.loads((R/'ground_lift_training_config.json').read_text());prep=json.loads((R/'baseline_configs/standby_preparation_config.json').read_text());assets=['basket_rectangular'];k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
    def visual_only(source,destination):
        if destination.exists():return
        import re
        src=Usd.Stage.Open(str(source))
        text=src.Flatten().ExportToString()
        text=text.replace('"Physics','"DisabledPhysics').replace('"Physx','"DisabledPhysx')
        text=re.sub(r'\bdef Physics\w+\s', 'def Xform ',text)
        layer=Sdf.Layer.CreateAnonymous('.usda');layer.ImportFromString(text);layer.Export(str(destination))
    C=R/'visibility_visual_assets';C.mkdir(exist_ok=True)
    seed=OLD/'collection_trials/collection_plans/placement_0023/case_handheld/candidate0_repeat1/prepared_scene.usdc'
    visual_only(seed,C/'robot_scene.usdc')
    for aid in assets:
        folder='strategy15_surface_assets' if aid in ['stool_rectangular','stool_wood_four_leg','table_side','basket_rectangular'] else 'object_assets'
        visual_only(R/folder/(aid+'.usdc'),C/(aid+'.usdc'))
    omni.usd.get_context().open_stage(str(C/'robot_scene.usdc'));stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer());omni.timeline.get_timeline_interface().stop()
    links={p.GetName():p for p in stage.Traverse() if str(p.GetPath()).startswith('/robot/') and p.GetName() in k.links}
    UsdGeom.Imageable(stage.GetPrimAtPath('/World/TestObject')).MakeInvisible();add_wood_floor(stage,R)
    targets={}
    for aid in assets:
        prim=stage.DefinePrim('/World/VisibilityTargets/'+aid,'Xform');prim.GetReferences().AddReference(str(C/(aid+'.usdc')));add_update_semantics(prim,'target');UsdGeom.Imageable(prim).MakeInvisible();targets[aid]=prim
    UsdPhysics.Scene.Define(stage,'/physicsScene')
    world=World(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/30)
    camera=Camera('/World/WristPreview',resolution=(1000,800));camera.initialize();camera._frequency=-1
    camera.set_focal_length(20.);camera.set_horizontal_aperture(32.);camera.set_vertical_aperture(24.);camera.set_clipping_range(.005,100.)
    from omni.isaac.core.utils.viewports import set_camera_view
    camera.set_focal_length(24.);camera.set_horizontal_aperture(28.);camera.set_vertical_aperture(22.4)
    aid='basket_rectangular';target=targets[aid];UsdGeom.Imageable(target).MakeVisible()
    for variant in ['repair_final_plans','mirror_right','mirror_home_seed']:
        p=json.loads((R/variant/(aid+'.json')).read_text())['plans'][0];q=p['q_grasp'];fk=k.fk(q,np.array(p['base_pose']))
        for name,prim in links.items():UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(fk[name].T.tolist()))
        t=np.eye(4);t[:3,3]=p['object_position'];UsdGeom.Xformable(target).MakeMatrixXform().Set(Gf.Matrix4d(t.T.tolist()))
        for view,eye,look in [('front',[2.1,0,.85],[.3,0,.58]),('top',[.35,.001,2.5],[.35,0,.3]),('oblique',[1.8,1.5,1.4],[.3,0,.5])]:
            set_camera_view(eye=np.array(eye),target=np.array(look),camera_prim_path='/World/WristPreview')
            for _ in range(3):world.render()
            rep.orchestrator.step(rt_subframes=4,pause_timeline=True);rgb=camera.get_rgba()[:,:,:3].astype(np.uint8);cv2.imwrite(str(O/(variant+'_'+view+'.jpg')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        print('PREVIEW',variant,flush=True)
except BaseException:
    import traceback;traceback.print_exc()
finally:app.close()
