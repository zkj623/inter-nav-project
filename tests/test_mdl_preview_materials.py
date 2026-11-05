import math
import numpy as np
import pytest
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
from grutopia_extension.interactive_navigation.mdl_preview_materials import convert_exported_pbr_materials,exported_uv_transform


@pytest.mark.parametrize('uva',[(1.,1.,0.,0.),(2.,3.,math.pi/2,0.),(.7,1.4,-.37,0.)])
def test_uv_transform_matches_exported_double_v_flip(uva):
    uv=np.array([.21,.83]);sx,sy,a=uva[:3]
    original=np.array([uv[0]*sx,(1-uv[1])*sy])-.5
    rotate=np.array([[math.cos(-a),-math.sin(-a)],[math.sin(-a),math.cos(-a)]])
    original=rotate@original+.5;original[1]=1-original[1]
    scale,degrees,offset=exported_uv_transform(uva);angle=math.radians(degrees)
    matrix=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    assert np.allclose(matrix@(uv*scale)+offset,original)


def make_material(tmp_path,translucent=False):
    name='Num5cc1227244c5ba2b1069f468.mdl' if translucent else 'Num5dd77cea7d6a630001bffad3.mdl'
    source=tmp_path/name
    source.write_text('OmniUe4 BaseColor_Color uniform texture_2d Normal_Tex = texture_2d("normal.png");')
    (tmp_path/'normal.png').write_bytes(b'test-image-resource');(tmp_path/'color.png').write_bytes(b'test-image-resource')
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/Scene')
    mat=UsdShade.Material.Define(stage,'/Scene/Material');shader=UsdShade.Shader.Define(stage,'/Scene/Material/Shader')
    shader.SetSourceAsset(Sdf.AssetPath(str(source)),'mdl');shader.SetSourceAssetSubIdentifier(name[:-4],'mdl')
    shader.CreateOutput('out',Sdf.ValueTypeNames.Token);mat.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
    return stage,mat,shader


def test_original_texture_and_uv_are_preserved_and_mdl_blocked(tmp_path):
    stage,mat,shader=make_material(tmp_path)
    shader.CreateInput('IsBaseColorTex',Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateInput('BaseColor_Tex',Sdf.ValueTypeNames.Asset).Set(str(tmp_path/'color.png'))
    shader.CreateInput('BaseColor_UVA',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.,3.,0.,0.))
    stats=convert_exported_pbr_materials(stage,'/Scene')
    tex=UsdShade.Shader(stage.GetPrimAtPath('/Scene/Material/Shader/BaseColorTexture'))
    uv=UsdShade.Shader(stage.GetPrimAtPath('/Scene/Material/Shader/BaseColorUV'))
    assert tex.GetInput('file').Get().resolvedPath==str(tmp_path/'color.png')
    assert tuple(uv.GetInput('translation').Get())==(0.,-2.)
    assert shader.GetInput('diffuseColor').HasConnectedSource() and stats['texture_resampling'] is False
    assert shader.GetPrim().GetAttribute('info:mdl:sourceAsset').Get() is None


def test_color_roughness_opacity_and_contact_properties_survive(tmp_path):
    stage,mat,shader=make_material(tmp_path,translucent=True)
    shader.CreateInput('BaseColor_Color',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.2,.3,.4,1.))
    shader.CreateInput('Gloss_Color',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.7,.7,.7,1.))
    shader.CreateInput('Opacity',Sdf.ValueTypeNames.Float).Set(.45)
    physical=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());physical.CreateStaticFrictionAttr(.6)
    convert_exported_pbr_materials(stage,'/Scene')
    assert tuple(shader.GetInput('diffuseColor').Get())==pytest.approx((.2,.3,.4))
    assert shader.GetInput('roughness').Get()==pytest.approx(.3)
    assert shader.GetInput('opacity').Get()==pytest.approx(.45)
    assert physical.GetStaticFrictionAttr().Get()==pytest.approx(.6)


def test_required_missing_texture_is_not_silently_replaced_by_color(tmp_path):
    stage,_,shader=make_material(tmp_path)
    shader.CreateInput('IsBaseColorTex',Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateInput('BaseColor_Tex',Sdf.ValueTypeNames.Asset).Set(str(tmp_path/'missing.png'))
    with pytest.raises(FileNotFoundError,match='BaseColor_Tex'):
        convert_exported_pbr_materials(stage,'/Scene')
