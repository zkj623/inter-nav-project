from pxr import Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
import pytest
from grutopia_extension.interactive_navigation.material_compaction import compact_scene_materials


def fixture(tmp_path):
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/Scene')
    source=tmp_path/'surface.mdl';source.write_text('mdl 1.4;')
    def material(name,roughness=.4):
        m=UsdShade.Material.Define(stage,'/Scene/'+name)
        shader=UsdShade.Shader.Define(stage,str(m.GetPath())+'/Shader')
        shader.SetSourceAsset(Sdf.AssetPath(str(source)),'mdl')
        shader.SetSourceAssetSubIdentifier('surface','mdl')
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateOutput('out',Sdf.ValueTypeNames.Token)
        m.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
        return m,shader
    return stage,material


def test_identical_materials_share_graph_with_same_geometry(tmp_path):
    stage,make=fixture(tmp_path);a,_=make('A');b,_=make('B')
    body=UsdGeom.Cube.Define(stage,'/Scene/Body')
    body.CreateSizeAttr(1.25);UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(b)
    report=compact_scene_materials(stage,'/Scene')
    bound,_=UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial()
    assert bound.GetPath()==a.GetPath()
    assert report['duplicate_materials_disabled']==1 and report['effective_bindings_verified']>0
    assert not b.GetPrim().IsActive() and body.GetSizeAttr().Get()==1.25
    assert body.GetPrim().HasAPI(UsdPhysics.CollisionAPI)


def test_different_shading_parameters_are_not_merged(tmp_path):
    stage,make=fixture(tmp_path);a,_=make('A',.4);b,_=make('B',.8)
    assert compact_scene_materials(stage,'/Scene')['duplicate_materials_disabled']==0
    assert a.GetPrim().IsActive() and b.GetPrim().IsActive()


def test_contact_material_and_inherited_binding_preserved(tmp_path):
    stage,make=fixture(tmp_path);a,_=make('A');b,_=make('B')
    physics=UsdPhysics.MaterialAPI.Apply(b.GetPrim());physics.CreateStaticFrictionAttr(.73)
    parent=UsdGeom.Xform.Define(stage,'/Scene/Parent').GetPrim()
    child=UsdGeom.Cube.Define(stage,'/Scene/Parent/Body').GetPrim()
    UsdShade.MaterialBindingAPI.Apply(parent).Bind(b,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    UsdShade.MaterialBindingAPI.Apply(child).Bind(a)
    report=compact_scene_materials(stage,'/Scene')
    material,_=UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial(materialPurpose='physics')
    assert material.GetPath()==b.GetPath() and report['physical_bindings_preserved']>0
    assert physics.GetStaticFrictionAttr().Get()==pytest.approx(.73)


def test_cross_material_shader_connections_keep_donor_alive(tmp_path):
    stage,make=fixture(tmp_path);make('A');donor,shader=make('B');_,consumer=make('Consumer')
    consumer.CreateInput('shared',Sdf.ValueTypeNames.Token).ConnectToSource(shader.ConnectableAPI(),'out')
    compact_scene_materials(stage,'/Scene')
    assert donor.GetPrim().IsActive()
    assert consumer.GetInput('shared').GetConnectedSource()[0].GetPrim().IsActive()


def test_unused_visual_graph_is_pruned_but_contact_material_is_retained(tmp_path):
    stage,make=fixture(tmp_path);used,_=make('Used',.3);unused,_=make('Unused',.8);contact,_=make('Contact',.6)
    UsdPhysics.MaterialAPI.Apply(contact.GetPrim()).CreateStaticFrictionAttr(.7)
    body=UsdGeom.Cube.Define(stage,'/Scene/Body').GetPrim();UsdShade.MaterialBindingAPI.Apply(body).Bind(used)
    report=compact_scene_materials(stage,'/Scene',prune_unused=True)
    assert report['unused_materials_disabled']==1
    assert not unused.GetPrim().IsActive() and contact.GetPrim().IsActive() and used.GetPrim().IsActive()
    bound,_=UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial()
    assert bound.GetPath()==used.GetPath()
