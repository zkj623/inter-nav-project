"""Cook all current catalog meshes without taking a physics step."""
import argparse,json,sys,traceback
from pathlib import Path
from catalog_support import DEFAULT_CATALOG, REPOSITORY, verify_catalog
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_CATALOG)
p.add_argument('--output',type=Path,required=True);p.add_argument('--collision-deps',type=Path)
p.add_argument('--gpu',type=int,default=0);p.add_argument('--only',default='');a=p.parse_args()
verify_catalog(a.root)
base=Path(__file__).resolve().parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(REPOSITORY)]
if a.collision_deps:sys.path.insert(0,str(a.collision_deps))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False})
from pxr import UsdGeom,UsdPhysics,PhysxSchema
import omni.usd,omni.physx
from cooked_geometry import CookedGeometry
from load_asset import add_asset
import carb
carb.settings.get_settings().set('/physics/cooking/ujitsoCollisionCooking',False)
manifest=json.loads((a.root/'asset_manifest.json').read_text())
if a.only:
    selected=set(a.only.split(','));known={r['id'] for r in manifest['assets']}
    if selected-known:raise ValueError('Unknown catalog IDs: '+str(sorted(selected-known)))
    manifest['assets']=[r for r in manifest['assets'] if r['id'] in selected]
rows=[]
physics_steps=[]
subscription=omni.physx.get_physx_interface().subscribe_physics_step_events(lambda dt:physics_steps.append(float(dt)))
try:
    for asset in manifest['assets']:
        try:
            omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
            UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
            UsdGeom.Xform.Define(stage,'/World');UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
            prim=add_asset(stage,asset['id'],'/World/Object',mass_kg=1.,static_friction=1.,dynamic_friction=.9,package_root=a.root)
            from pxr import Usd
            for child in Usd.PrimRange(prim):
                if child.IsA(UsdGeom.Mesh) and child.HasAPI(UsdPhysics.CollisionAPI):
                    if UsdGeom.Imageable(child).ComputeVisibility()==UsdGeom.Tokens.invisible:
                        UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
                    else:
                        c=PhysxSchema.PhysxCollisionAPI.Apply(child)
                        c.CreateContactOffsetAttr(.002);c.CreateRestOffsetAttr(.0001)
            # Collision meshes share identity transforms in body-local metres.
            from bake_collision_meshes import bake_collision_meshes
            bake_collision_meshes(stage,prim,out/(asset['id']+'_bake.json'))
            omni.physx.get_physx_interface().force_load_physics_from_usd()
            g=CookedGeometry.from_stage(stage,'/World/Object',asset['dimensions_m'],out/(asset['id']+'.json'))
            row=dict(asset_id=asset['id'],geometry_ready=True,hulls=len(g.parts),sha256=g.sha256)
            from grutopia_extension.manipulation.household_objects import grasp_height
            try:row['profile']=g.grasp_profile(grasp_height('box',asset['dimensions_m']))
            except Exception as exc:row['profile_error']=str(exc)
        except Exception as exc:
            row=dict(asset_id=asset['id'],geometry_ready=False,error=str(exc),traceback=traceback.format_exc())
        rows.append(row);print(json.dumps(row),flush=True)
        (out/'summary.json').write_text(json.dumps(rows,indent=2))
finally:
    (out/'metadata.json').write_text(json.dumps(dict(physics_steps=len(physics_steps),assets=len(rows),
        all_geometry_ready=len(rows)==len(manifest['assets']) and all(r['geometry_ready'] for r in rows)),indent=2))
    app.close()
