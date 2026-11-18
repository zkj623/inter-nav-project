"""Expand the pinned official G2 Xacro without requiring a ROS installation."""
from pathlib import Path
import sys, json, subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'python-deps'))
import xacro
import xacro.substitution_args
vendor = ROOT / 'vendor/genie_sim_robot_model'
def package_path(name):
    if name != 'genie_sim_robot_model':
        raise ValueError('Unexpected ROS package: ' + name)
    return str(vendor)
xacro.substitution_args._eval_find = package_path
doc = xacro.process_file(str(vendor / 'xacro/robot.xacro'), mappings={
    'robot_model':'genie', 'body':'g2', 'arm':'crs', 'gripper':'omnipicker'})
text = doc.toprettyxml().replace('package://genie_sim_robot_model/', str(vendor) + '/')
out = ROOT / 'generated'
out.mkdir(exist_ok=True)
(out / 'g2_crs_omnipicker.urdf').write_text(text)
robot = ET.fromstring(text)
missing = [n.attrib['filename'] for n in robot.findall('.//mesh') if not Path(n.attrib['filename']).is_file()]
report = {'source_commit':subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip(),
          'configuration':{'body':'g2','arm':'crs','gripper':'omnipicker'},
          'links':len(robot.findall('link')), 'joints':len(robot.findall('joint')),
          'missing_meshes':missing,
          'movable_joints':[{'name':j.attrib['name'],'type':j.attrib['type'],
                             'limit': j.find('limit').attrib if j.find('limit') is not None else {},
                             'mimic': j.find('mimic').attrib if j.find('mimic') is not None else None}
                            for j in robot.findall('joint') if j.attrib['type']!='fixed']}
(out / 'model_manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
assert not missing, missing
