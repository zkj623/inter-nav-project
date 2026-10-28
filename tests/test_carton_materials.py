import unittest
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
from grutopia_extension.interactive_navigation.material_fallback import apply_simple_scene_material
from grutopia_extension.interactive_navigation.carton_materials import restore_carton_materials


class CartonMaterialTests(unittest.TestCase):
    def setUp(self):
        self.stage=Usd.Stage.CreateInMemory();self.root='/World/env_0/scene'
        UsdGeom.Xform.Define(self.stage,self.root)
        UsdGeom.Xform.Define(self.stage,self.root+'/InteractiveBoxes/B01')
        self.body=UsdGeom.Cube.Define(self.stage,self.root+'/InteractiveBoxes/B01/Body').GetPrim()
        self.floor=UsdGeom.Cube.Define(self.stage,self.root+'/Floor').GetPrim()
        UsdPhysics.CollisionAPI.Apply(self.body)
        self.material=UsdShade.Material.Define(self.stage,self.root+'/BoxMaterials/B01_Cardboard')
        physics=UsdPhysics.MaterialAPI.Apply(self.material.GetPrim())
        physics.CreateStaticFrictionAttr(.6);physics.CreateDynamicFrictionAttr(.5);physics.CreateRestitutionAttr(0.)
        binding=UsdShade.MaterialBindingAPI.Apply(self.body)
        binding.Bind(self.material);binding.Bind(self.material,materialPurpose='physics')

    def test_visual_simplification_then_restore_preserves_friction(self):
        apply_simple_scene_material(self.stage,self.root)
        preserved,_=UsdShade.MaterialBindingAPI(self.body).ComputeBoundMaterial(materialPurpose='physics')
        self.assertEqual(preserved.GetPath(),self.material.GetPath())
        self.assertTrue(self.material.GetPrim().IsActive())
        audit=restore_carton_materials(self.stage,self.root,['B01'])
        bound,_=UsdShade.MaterialBindingAPI(self.body).ComputeBoundMaterial(materialPurpose='physics')
        self.assertEqual(bound.GetPath(),self.material.GetPath())
        self.assertAlmostEqual(audit[0]['static_friction'],.6,places=6)
        self.assertAlmostEqual(audit[0]['dynamic_friction'],.5,places=6)
        self.assertEqual(audit[0]['colliders'],1)

    def test_restore_keeps_other_scene_geometry_simplified(self):
        apply_simple_scene_material(self.stage,self.root)
        restore_carton_materials(self.stage,self.root,['B01'])
        bound,_=UsdShade.MaterialBindingAPI(self.floor).ComputeBoundMaterial()
        self.assertEqual(str(bound.GetPath()),self.root+'/NavigationSimpleMaterial')

    def test_all_purpose_and_inherited_contact_materials_are_preserved(self):
        parent=UsdGeom.Xform.Define(self.stage,self.root+'/Furniture').GetPrim()
        child=UsdGeom.Cube.Define(self.stage,self.root+'/Furniture/Part').GetPrim()
        UsdPhysics.CollisionAPI.Apply(child)
        UsdShade.MaterialBindingAPI.Apply(parent).Bind(self.material)
        UsdShade.MaterialBindingAPI.Apply(self.floor).Bind(self.material)
        apply_simple_scene_material(self.stage,self.root)
        for prim in [child,self.floor]:
            bound,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            self.assertEqual(bound.GetPath(),self.material.GetPath())
            physics=UsdPhysics.MaterialAPI(bound.GetPrim())
            self.assertAlmostEqual(physics.GetStaticFrictionAttr().Get(),.6,places=6)
            self.assertAlmostEqual(physics.GetDynamicFrictionAttr().Get(),.5,places=6)

    def test_physical_material_survives_while_visual_graph_is_disabled(self):
        shader=UsdShade.Shader.Define(self.stage,str(self.material.GetPath())+'/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        self.material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        apply_simple_scene_material(self.stage,self.root)
        self.assertTrue(self.material.GetPrim().IsActive())
        self.assertFalse(shader.GetPrim().IsActive())
        restore_carton_materials(self.stage,self.root,['B01'])
        self.assertTrue(shader.GetPrim().IsActive())


    def test_nested_visual_graph_can_be_disabled_and_restored(self):
        graph=UsdShade.NodeGraph.Define(self.stage,str(self.material.GetPath())+'/Network')
        shader=UsdShade.Shader.Define(self.stage,str(graph.GetPath())+'/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader_path=shader.GetPath();graph_path=graph.GetPath()
        apply_simple_scene_material(self.stage,self.root)
        self.assertFalse(self.stage.GetPrimAtPath(graph_path).IsActive())
        restore_carton_materials(self.stage,self.root,['B01'])
        self.assertTrue(self.stage.GetPrimAtPath(graph_path).IsActive())
        self.assertTrue(self.stage.GetPrimAtPath(shader_path).IsActive())


if __name__=='__main__':unittest.main()
