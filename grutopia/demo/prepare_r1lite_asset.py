"""Import official Galaxea R1 Lite meshes and restore its physical swerve base."""

import argparse, copy, hashlib, json, os
from pathlib import Path
from xml.etree import ElementTree as E

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
source = Path('/data20t/embodied/zkj/third_party/GalaxeaManipSim/galaxea_sim/assets/r1_lite')
reference = Path('/data20t/embodied/zkj/third_party/galaxea_urdf/R1Lite/urdf_r1lite_2025.urdf')
r = E.parse(source / 'robot.urdf')
official = {j.attrib['name']: j for j in E.parse(reference).getroot().findall('joint')}
restored = []
for j in r.getroot().findall('joint'):
    if j.attrib['name'].startswith(('wheel_motor_joint', 'steer_motor_joint')):
        original = official[j.attrib['name']]
        j.attrib['type'] = original.attrib['type']
        limit = j.find('limit')
        if limit is not None:
            j.remove(limit)
        if original.find('limit') is not None:
            j.append(copy.deepcopy(original.find('limit')))
        # Official steering uses +/-pi. For exploration allow unlimited swivel;
        # the physical hardware supports full rotation, targets stay near current angle.
        if j.attrib['name'].startswith('steer_motor_joint'):
            j.attrib['type'] = 'continuous'
        restored.append(j.attrib['name'])
# Upstream manipulation collision wheels are shrunk to ~47 mm radius,
# while their visual tires are 70 mm. Restore actual cylindrical contact
# geometry so odometry and wheel inverse kinematics use the same radius.
for link in r.getroot().findall('link'):
    if link.attrib['name'].startswith('wheel_motor_link'):
        for collider in list(link.findall('collision')):
            link.remove(collider)
        collider = E.SubElement(link, 'collision')
        E.SubElement(collider, 'origin', xyz='0 0 0', rpy='1.5707963267948966 0 0')
        geometry = E.SubElement(collider, 'geometry')
        E.SubElement(geometry, 'cylinder', radius='0.07', length='0.048')
for m in r.getroot().findall('.//mesh'):
    resolved = source / m.attrib['filename']
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    m.attrib['filename'] = str(resolved)
urdf = out / 'r1lite_navigation.urdf'
r.write(urdf, encoding='utf-8', xml_declaration=True)
from isaacsim import SimulationApp

app = SimulationApp({'headless': True, 'active_gpu': a.gpu, 'physics_gpu': a.gpu, 'multi_gpu': False})
import omni.kit.commands
from omni.isaac.core.utils.extensions import enable_extension

enable_extension('omni.importer.urdf')
from omni.importer.urdf import _urdf

cfg = _urdf.ImportConfig()
cfg.merge_fixed_joints = False
cfg.fix_base = False
cfg.make_default_prim = True
cfg.create_physics_scene = False
cfg.import_inertia_tensor = True
cfg.self_collision = False
cfg.default_drive_strength = 1000
cfg.default_position_drive_damping = 100
cfg.distance_scale = 1.0
usd = out / 'r1lite_navigation.usd'
ok, root = omni.kit.commands.execute(
    'URDFParseAndImportFile', urdf_path=str(urdf), import_config=cfg, dest_path=str(usd)
)
if not ok or not usd.exists():
    raise RuntimeError('R1 Lite URDF import failed')
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open(str(usd))
d = stage.GetDefaultPrim()
joints = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
manifest = {
    'platform': 'Galaxea R1 Lite',
    'source_urdf': str(source / 'robot.urdf'),
    'wheel_joint_reference': str(reference),
    'wheel_contact_geometry': {
        'shape': 'cylinder',
        'radius_m': 0.07,
        'width_m': 0.048,
        'reason': 'upstream visual tire dimension; manipulation collider was ~.047 m radius',
    },
    'restored_joints': restored,
    'default_prim': str(d.GetPath()),
    'usd_sha256': hashlib.sha256(usd.read_bytes()).hexdigest(),
    'source_license': 'GalaxeaManipSim/LICENSE: Apache-2.0; public reference URDF has no separate LICENSE',
    'mesh_sources': str(source / 'meshes'),
    'revolute_joints': joints,
    'scale': 1.0,
}
(out / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest), flush=True)
app.close()
