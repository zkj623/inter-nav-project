"""Share byte-identical image assets while keeping texture-node sampling settings."""
import hashlib
import json
from pathlib import Path


def compact_texture_assets(stage,root_path):
    from pxr import Usd,UsdShade
    cache={};canonical={};mapping=[];rewritten=0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdShade.Shader):continue
        shader=UsdShade.Shader(prim)
        if shader.GetIdAttr().Get()!='UsdUVTexture':continue
        inp=shader.GetInput('file');asset=inp.Get() if inp else None
        if asset is None:continue
        if not asset.resolvedPath:raise FileNotFoundError(f'Unresolved texture: {asset.path}')
        path=Path(asset.resolvedPath)
        if path not in cache:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            cache[path]=(h.hexdigest(),path.suffix.lower())
        key=cache[path];shared=canonical.setdefault(key,path)
        if shared!=path:
            inp.Set(str(shared));rewritten+=1
        mapping.append((str(prim.GetPath()),str(path),str(shared),key[0]))
    return {'source_texture_paths':len(cache),'unique_texture_contents':len(canonical),
        'file_connections_rewritten':rewritten,'image_bytes_unchanged':True,'texture_resampling':False,
        'mapping_sha256':hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()}
