"""A visual-only household wood floor; preserves original collision geometry."""
from pathlib import Path
import json
def add_wood_floor(stage, root):
    from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf
    root=Path(root)
    provenance=json.loads((root/'home_floor_assets/wood_floor/provenance.json').read_text())
    hidden=[]
    for prim in list(stage.Traverse()):
        path=str(prim.GetPath());low=path.lower()
        if prim.IsA(UsdGeom.Gprim) and ('ground' in low or 'floor' in low) and not path.startswith('/robot/'):
            UsdGeom.Imageable(prim).MakeInvisible();hidden.append(path)
    material=UsdShade.Material.Define(stage,'/World/HomeFloorLooks/Wood')
    shader=UsdShade.Shader.Define(stage,'/World/HomeFloorLooks/Wood/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    shader.CreateInput('specular',Sdf.ValueTypeNames.Float).Set(.3)
    reader=UsdShade.Shader.Define(stage,'/World/HomeFloorLooks/Wood/UV')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    for key,slot,channel,value_type in [('diffuse','diffuseColor','rgb',Sdf.ValueTypeNames.Color3f),('rough','roughness','r',Sdf.ValueTypeNames.Float),('nor_gl','normal','rgb',Sdf.ValueTypeNames.Normal3f)]:
        tex=UsdShade.Shader.Define(stage,'/World/HomeFloorLooks/Wood/'+key)
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(provenance['maps'][key]['path'])
        tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB' if key=='diffuse' else 'raw')
        tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.GetOutput('result'))
        if key=='nor_gl':
            tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2,2,2,1))
            tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1,-1,-1,0))
        tex.CreateOutput(channel,Sdf.ValueTypeNames.Float3 if channel=='rgb' else Sdf.ValueTypeNames.Float)
        shader.CreateInput(slot,value_type).ConnectToSource(tex.GetOutput(channel))
    shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput('surface'))
    mesh=UsdGeom.Mesh.Define(stage,'/World/HouseholdFloorVisual')
    mesh.CreatePointsAttr([(-10,-10,.0005),(10,-10,.0005),(10,10,.0005),(-10,10,.0005)])
    mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    mesh.CreateNormalsAttr([(0,0,1)]*4);mesh.SetNormalsInterpolation('vertex')
    mesh.CreateSubdivisionSchemeAttr('none')
    size=provenance['physical_texture_width_m']
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set([(-10/size,-10/size),(10/size,-10/size),(10/size,10/size),(-10/size,10/size)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return dict(type='household wood floor',source=provenance['source'],license=provenance['license'],texture_repeat_m=size,texture_maps=provenance['maps'],hidden_visual_prims=hidden,physics_surface_unchanged=True,visual_only_plane_z_m=.0005,displacement=False)
