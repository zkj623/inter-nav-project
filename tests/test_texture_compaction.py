import pytest
from pxr import Sdf,Usd,UsdGeom,UsdShade
from grutopia_extension.interactive_navigation.texture_compaction import compact_texture_assets


def texture(stage,name,path,color_space):
    shader=UsdShade.Shader.Define(stage,'/Scene/'+name)
    shader.CreateIdAttr('UsdUVTexture');shader.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(path))
    shader.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set(color_space)
    return shader


def test_identical_image_bytes_share_file_without_changing_color_space(tmp_path):
    a=tmp_path/'one.png';b=tmp_path/'two.png';a.write_bytes(b'encoded-image');b.write_bytes(a.read_bytes())
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/Scene')
    first=texture(stage,'A',a,'sRGB');second=texture(stage,'B',b,'raw')
    report=compact_texture_assets(stage,'/Scene')
    assert report['unique_texture_contents']==1 and report['file_connections_rewritten']==1
    assert second.GetInput('file').Get().resolvedPath==first.GetInput('file').Get().resolvedPath
    assert second.GetInput('sourceColorSpace').Get()=='raw' and first.GetInput('sourceColorSpace').Get()=='sRGB'
    assert a.read_bytes()==b.read_bytes()==b'encoded-image'


def test_different_image_content_is_not_merged(tmp_path):
    a=tmp_path/'one.png';b=tmp_path/'two.png';a.write_bytes(b'first-image');b.write_bytes(b'different-image')
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/Scene')
    texture(stage,'A',a,'sRGB');second=texture(stage,'B',b,'sRGB')
    report=compact_texture_assets(stage,'/Scene')
    assert report['unique_texture_contents']==2 and report['file_connections_rewritten']==0
    assert second.GetInput('file').Get().resolvedPath==str(b)


def test_missing_image_is_reported_instead_of_dropped(tmp_path):
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/Scene')
    texture(stage,'A',tmp_path/'missing.png','sRGB')
    with pytest.raises(FileNotFoundError):compact_texture_assets(stage,'/Scene')
