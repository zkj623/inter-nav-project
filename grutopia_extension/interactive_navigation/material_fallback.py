"""Runtime UsdPreviewSurface fallbacks for unsupported GRScenes KooPbr MDLs."""

import re
from pathlib import Path

import numpy as np

_COLOR_PATTERN = re.compile(
    r'diffuse\s*:\s*color\(\s*([-+0-9.eE]+)f?\s*,\s*'
    r'([-+0-9.eE]+)f?\s*,\s*([-+0-9.eE]+)f?\s*\)'
)
_TEXTURE_PATTERN = re.compile(r'diffuse\s*:.*?texture_2d\("([^"]+)"', re.DOTALL)


def _scene_prims(stage, root_path):
    from pxr import Usd

    if root_path is None:
        return stage.Traverse()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f'Scene root not found: {root_path}')
    return Usd.PrimRange(root)


def apply_simple_scene_material(stage, root_path) -> dict:
    """Replace only the referenced scene's materials in the runtime layer.

    This removes texture appearance; use for geometry tests, not visual-model
    quality evaluation. Robot, sensors, physics and semantic labels are kept.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    material_path = Sdf.Path(root_path).AppendChild('NavigationSimpleMaterial')
    materials = [
        prim for prim in _scene_prims(stage, root_path)
        if prim.IsA(UsdShade.Material) and prim.GetPath() != material_path
    ]
    # Disable original shader networks as well as their bindings so Hydra does
    # not discover and compile their MDL/texture assets on the first update.
    for prim in materials:
        prim.SetActive(False)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Shader'))
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    geometries = 0
    for prim in _scene_prims(stage, root_path):
        if prim.IsA(UsdGeom.Gprim):
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.UnbindAllBindings()
            binding.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            geometries += 1
    return {'disabled_materials': len(materials), 'geometries': geometries}


def apply_koostruct_material_fallbacks(stage, root_path=None) -> dict:
    from pxr import Gf, Sdf, UsdShade

    replaced = 0
    textured = 0
    colored = 0
    missing_sources = 0
    for prim in _scene_prims(stage, root_path):
        source_attribute = prim.GetAttribute('info:mdl:sourceAsset')
        if not source_attribute:
            continue
        source_asset = source_attribute.Get()
        source_path = _resolved_asset_path(source_asset)
        if source_path is None or not source_path.exists():
            missing_sources += 1
            continue
        source = source_path.read_text(encoding='utf-8', errors='ignore')
        is_day_material = source_path.name == 'DayMaterial.mdl'
        if '::KooPbr' not in source and not is_day_material:
            continue

        shader = UsdShade.Shader(prim)
        shader.CreateIdAttr('UsdPreviewSurface')
        prim.GetAttribute('info:implementationSource').Set('id')
        source_attribute.Clear()
        sub_identifier = prim.GetAttribute('info:mdl:sourceAsset:subIdentifier')
        if sub_identifier:
            sub_identifier.Clear()

        diffuse_input = shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)
        texture_path = None if is_day_material else _texture_path(source_path, source)
        if texture_path is not None:
            _connect_texture(stage, prim.GetPath(), diffuse_input, texture_path)
            textured += 1
        else:
            diffuse_input.Set(Gf.Vec3f(*(1.0, 1.0, 1.0) if is_day_material else _diffuse_color(source)))
            colored += 1
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(_roughness(source))
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(_metallic(source))
        shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)

        material = UsdShade.Material(prim.GetParent())
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        mdl_output = material.GetOutput('mdl:surface')
        if mdl_output:
            mdl_output.DisconnectSource()
        replaced += 1
    return {
        'replaced': replaced,
        'textured': textured,
        'colored': colored,
        'missing_sources': missing_sources,
    }


def _resolved_asset_path(asset) -> Path | None:
    if asset is None:
        return None
    resolved = getattr(asset, 'resolvedPath', '')
    if not resolved:
        return None
    return Path(resolved)


def _diffuse_color(source: str):
    match = _COLOR_PATTERN.search(source)
    if match is None:
        return 0.55, 0.55, 0.55
    return tuple(float(value) for value in np.clip([float(value) for value in match.groups()], 0.0, 1.0))


def _texture_path(source_path: Path, source: str) -> Path | None:
    match = _TEXTURE_PATTERN.search(source)
    if match is None:
        return None
    texture_path = (source_path.parent / match.group(1)).resolve()
    return texture_path if texture_path.is_file() else None


def _roughness(source: str) -> float:
    match = re.search(r'reflect_glossiness\s*:\s*([-+0-9.eE]+)f?', source)
    if match is None:
        return 0.5
    return float(np.clip(1.0 - float(match.group(1)), 0.05, 1.0))


def _metallic(source: str) -> float:
    match = re.search(r'reflection_metalness\s*:\s*([-+0-9.eE]+)f?', source)
    return 0.0 if match is None else float(np.clip(float(match.group(1)), 0.0, 1.0))


def _connect_texture(stage, shader_path, diffuse_input, texture_path: Path):
    from pxr import Sdf, UsdShade

    reader = UsdShade.Shader.Define(stage, shader_path.AppendChild('FallbackPrimvarReader'))
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, shader_path.AppendChild('FallbackTexture'))
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(texture_path))
    texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(),
        'result',
    )
    texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    diffuse_input.ConnectToSource(texture.ConnectableAPI(), 'rgb')
