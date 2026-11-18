from pathlib import Path
import json,hashlib
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':0,'multi_gpu':False})
from pxr import Usd
from inertia_from_urdf import apply_inertials
R=Path(__file__).resolve().parent;source=R/'robot_assets/g2_contact_explicit.usdc';out=source.with_name('g2_contact_inertia_fixed.usdc')
assert not out.exists()
stage=Usd.Stage.Open(str(source));rows=apply_inertials(stage,R/'generated/g2_crs_omnipicker.urdf');stage.GetRootLayer().Export(str(out));assert Usd.Stage.Open(str(out))
report=dict(source=str(source),output=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),rigid_bodies=len(rows),scope='URDF inertial frame restoration only; no drive strength or geometry changes')
(R/'inertia_fixed_asset.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));app.close()
