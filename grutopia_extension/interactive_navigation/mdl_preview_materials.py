"""Preview-surface compatibility for the two exported GRScenes UE4 PBR templates."""
import math
import re
from pathlib import Path

TEMPLATES={'Num5dd77cea7d6a630001bffad3.mdl','Num5cc1227244c5ba2b1069f468.mdl'}


def exported_uv_transform(uva):
    """Equivalent affine UV transform after the exporter's two V flips."""
    sx,sy,angle=map(float,uva[:3]);c,s=math.cos(angle),math.sin(angle)
    return (sx,sy),math.degrees(angle),(.5-.5*c-(.5-sy)*s,.5-.5*s+(.5-sy)*c)


def convert_exported_pbr_materials(stage,root_path):
    from pxr import Gf,Sdf,Usd,UsdShade
    root=stage.GetPrimAtPath(root_path)
    converted=textures=normal_maps=0;texture_paths=set()
    for prim in list(Usd.PrimRange(root)):
        asset_attr=prim.GetAttribute('info:mdl:sourceAsset')
        value=asset_attr.Get() if asset_attr else None
        source=Path(value.resolvedPath) if value and value.resolvedPath else None
        if source is None or source.name not in TEMPLATES:continue
        shader=UsdShade.Shader(prim)
        if shader.GetImplementationSourceAttr().Get()!='sourceAsset':continue
        text=source.read_text(encoding='utf-8-sig')
        if 'OmniUe4' not in text or 'BaseColor_Color' not in text:
            raise ValueError(f'Unexpected PBR template: {source}')
        def scalar(name,fallback):
            inp=shader.GetInput(name)
            if inp and inp.Get() is not None:return float(inp.Get())
            match=re.search(r'\bfloat\s+'+re.escape(name)+r'\s*=\s*([-+0-9.eE]+)',text)
            return float(match.group(1)) if match else fallback
        def vector(name,fallback):
            inp=shader.GetInput(name)
            if inp and inp.Get() is not None:return tuple(map(float,inp.Get()))
            match=re.search(r'\bfloat4\s+'+re.escape(name)+r'\s*=\s*float4\(([^)]+)\)',text)
            if not match:return fallback
            values=tuple(float(x.strip()) for x in match.group(1).split(','))
            return values*4 if len(values)==1 else values
        def texture_file(name):
            inp=shader.GetInput(name);asset=inp.Get() if inp else None
            if asset and asset.resolvedPath:return Path(asset.resolvedPath)
            if asset and asset.path:
                for candidate in [source.parent/asset.path,source.parent.parent/asset.path]:
                    if candidate.is_file():return candidate.resolve()
            match=re.search(r'\btexture_2d\s+'+re.escape(name)+r'\s*=\s*texture_2d\("([^"]+)"',text)
            if match:
                candidate=source.parent/match.group(1)
                if candidate.is_file():return candidate.resolve()
            raise FileNotFoundError(f'Material texture {name} at {prim.GetPath()}')
        # Read source parameters before replacing the implementation.
        specifications=[]
        for channel,output,kind,default in [
            ('BaseColor','diffuseColor','color',(1.,1.,1.,1.)),
            ('Metallic','metallic','scalar',(0.,0.,0.,1.)),
            ('Gloss','roughness','inverse',(0.1,0.1,0.1,1.)),
            ('Emissive','emissiveColor','emissive',(1.,1.,1.,1.))]:
            flag='Is'+channel+'Tex'
            enabled=abs(scalar(flag,0.)-1.)<=1e-5
            gain=scalar('EmissiveIntensity',0.) if kind=='emissive' else 1.
            specifications.append((channel,output,kind,vector(channel+'_Color',default),
                texture_file(channel+'_Tex') if enabled and gain!=0 else None,
                vector(channel+'_UVA',(1.,1.,0.,0.)),gain))
        normal=texture_file('Normal_Tex');normal_uva=vector('Normal_UVA',(1.,1.,0.,0.))
        opacity=scalar('Opacity',1.) if source.name=='Num5cc1227244c5ba2b1069f468.mdl' else 1.
        specular=vector('Specular_Color',(1.,1.,1.,1.))[0]
        shader.CreateIdAttr('UsdPreviewSurface');shader.GetImplementationSourceAttr().Set('id')
        asset_attr.Block()
        sub=prim.GetAttribute('info:mdl:sourceAsset:subIdentifier')
        if sub:sub.Block()
        reader=UsdShade.Shader.Define(stage,prim.GetPath().AppendChild('PreviewUV'))
        reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
        def connect(name,output,path,uva,scale,bias,color_space,scalar_output=False):
            nonlocal textures
            transform=UsdShade.Shader.Define(stage,prim.GetPath().AppendChild(name+'UV'))
            transform.CreateIdAttr('UsdTransform2d')
            transform.CreateInput('in',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            uv_scale,rotation,translation=exported_uv_transform(uva)
            transform.CreateInput('scale',Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*uv_scale))
            transform.CreateInput('rotation',Sdf.ValueTypeNames.Float).Set(rotation)
            transform.CreateInput('translation',Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*translation))
            transform.CreateOutput('result',Sdf.ValueTypeNames.Float2)
            tex=UsdShade.Shader.Define(stage,prim.GetPath().AppendChild(name+'Texture'))
            tex.CreateIdAttr('UsdUVTexture');tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(path))
            tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set(color_space)
            tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat');tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(),'result')
            tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
            component='r' if scalar_output else 'rgb'
            tex.CreateOutput(component,Sdf.ValueTypeNames.Float if scalar_output else Sdf.ValueTypeNames.Float3)
            output.ConnectToSource(tex.ConnectableAPI(),component)
            textures+=1;texture_paths.add(str(path))
        for channel,name,kind,color,path,uva,gain in specifications:
            scalar_output=kind in ['scalar','inverse']
            output=shader.CreateInput(name,Sdf.ValueTypeNames.Float if scalar_output else Sdf.ValueTypeNames.Color3f)
            if path:
                scale=(-1.,-1.,-1.,1.) if kind=='inverse' else (gain,gain,gain,1.)
                bias=(1.,1.,1.,0.) if kind=='inverse' else (0.,0.,0.,0.)
                connect(channel,output,path,uva,scale,bias,'sRGB',scalar_output)
            elif scalar_output:output.Set(1.-color[0] if kind=='inverse' else color[0])
            else:output.Set(Gf.Vec3f(*(x*gain for x in color[:3])))
        connect('Normal',shader.CreateInput('normal',Sdf.ValueTypeNames.Normal3f),normal,normal_uva,
            (2.,2.,2.,1.),(-1.,-1.,-1.,0.),'raw')
        normal_maps+=1
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(opacity)
        # UE4 scalar specular sets dielectric F0 to 0.08 * specular.
        f0=min(.99,max(0.,.08*specular));root_f0=math.sqrt(f0)
        shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set((1.+root_f0)/(1.-root_f0))
        shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
        material=UsdShade.Material(prim.GetParent())
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        mdl=material.GetOutput('mdl:surface')
        if mdl:mdl.DisconnectSource()
        converted+=1
    return {'converted':converted,'texture_connections':textures,'normal_maps':normal_maps,
        'unique_texture_files':len(texture_paths),'texture_resampling':False,
        'scope':'Exported base colors/textures/UVs, metallic, roughness, emission, normal and opacity retained in preview shading; refraction/displacement and textured specular are approximations.'}
