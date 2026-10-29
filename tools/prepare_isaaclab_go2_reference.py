"""Fetch pinned NVIDIA Isaac Lab Go2 references and validate a packed asset.

These are NVIDIA-published training references, not Unitree factory firmware.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request

ROOT='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab'
FILES={
    'go2.usd':(ROOT+'/Robots/Unitree/Go2/go2.usd','ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d'),
    'Props/instanceable_meshes.usd':(ROOT+'/Robots/Unitree/Go2/Props/instanceable_meshes.usd','2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5'),
    'rough_checkpoint.pt':(ROOT+'/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt','f2aa77bf0349c10ec2a00071162e37123d0cf80f3447c3989c17a618e1b24ad2'),
    'flat_checkpoint.pt':(ROOT+'/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt','47ccb2046ec3064257997f879c62d64217b937713c30d827883b2c2f40c50439'),
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(path,url,expected):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if digest(path)!=expected:raise RuntimeError(f'Refusing to overwrite mismatched existing file: {path}')
        return
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Go2-reference-preparation'}),timeout=30) as response:
        data=response.read()
    if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError(f'Unexpected downloaded hash: {url}')
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False,prefix='.go2-download-') as temporary:
        temporary.write(data);name=temporary.name
    os.replace(name,path)


def inspect_asset(path):
    from pxr import Usd,UsdGeom,UsdPhysics
    stage=Usd.Stage.Open(str(path));assert stage and stage.GetDefaultPrim()
    bodies=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    masses=[UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in bodies]
    colliders=sum(p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Gprim)
        and p.GetAttribute('physics:collisionEnabled').Get() is not False
        for p in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()))
    assert len(bodies)==19 and colliders==27 and UsdGeom.GetStageMetersPerUnit(stage)==1.
    assert all(m is not None and m>0 for m in masses) and abs(sum(masses)-15.019)<.01
    return stage,{'body_count':len(bodies),'collider_count':int(colliders),'declared_mass_kg':sum(masses),'meters_per_unit':1.}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-flat',action='store_true')
    args=parser.parse_args();out=args.output.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    names=list(FILES) if args.include_flat else [k for k in FILES if not k.startswith('flat_')]
    for name in names:fetch(out/name,*FILES[name])
    stage,structure=inspect_asset(out/'go2.usd')
    flat=out/'go2_training_flattened.usdc';manifest_path=out/'go2_reference_manifest.json'
    if flat.exists():
        old=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if old.get('packed_asset_sha256')!=digest(flat):raise RuntimeError('Existing packed asset has no matching provenance')
    else:
        with tempfile.NamedTemporaryFile(dir=out,suffix='.usdc',prefix='.go2-flatten-',delete=False) as temporary:name=temporary.name
        try:
            assert stage.Flatten().Export(name)
            inspect_asset(name)
            os.replace(name,flat)
        finally:
            if Path(name).exists():Path(name).unlink()
    inspect_asset(flat)
    report={'publisher':'NVIDIA Isaac Lab','not_unitree_factory_policy':True,
        'source_asset_release':'Isaac4.5','training_code_reference':'IsaacLab v2.0.0',
        'files':{name:{'url':FILES[name][0],'sha256':FILES[name][1],'path':str(out/name)} for name in names},
        'packed_asset':str(flat),'packed_asset_sha256':digest(flat),'structure':structure,
        'rough_policy':str(out/'rough_checkpoint.pt'),
        'note':'Flattening resolves references only; no geometry, scale, mass or joint transform edits.'}
    manifest_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
