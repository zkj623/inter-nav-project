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
        self.assertFalse(self.body.GetRelationship('material:binding:physics').GetTargets())
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


if __name__=='__main__':unittest.main()
