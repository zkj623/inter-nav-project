"""Unitree Go2 asset resolution: official USD/meshes, then primitive fallback."""

import math
from pathlib import Path
from typing import Mapping, Sequence
from xml.etree import ElementTree as ET

GO2_DEFAULT_JOINT_POSITIONS = (
    0.1,
    -0.1,
    0.1,
    -0.1,
    0.8,
    0.8,
    1.0,
    1.0,
    -1.5,
    -1.5,
    -1.5,
    -1.5,
)

OFFICIAL_GO2_USD_CANDIDATES = (
    'https://omniverse-content-production.s3-us-west-2.amazonaws.com/'
    'Assets/Isaac/4.2/Isaac/Robots/Unitree/Go2/go2.usd',
    'https://omniverse-content-production.s3-us-west-2.amazonaws.com/'
    'Assets/Isaac/4.5/Isaac/Robots/Unitree/Go2/go2.usd',
    'https://omniverse-content-production.s3-us-west-2.amazonaws.com/'
    'Assets/Isaac/4.2/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd',
    'https://omniverse-content-production.s3-us-west-2.amazonaws.com/'
    'Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd',
    'https://omniverse-content-production.s3-us-west-2.amazonaws.com/'
    'Assets/Isaac/5.0/Isaac/Robots/Unitree/Go2/go2.usd',
)
_MIN_ISAACLAB_USD_BYTES = 1_000_000
OFFICIAL_GO2_MESH_NAMES = (
    'base.dae',
    'hip.dae',
    'thigh.dae',
    'thigh_mirror.dae',
    'calf.dae',
    'calf_mirror.dae',
    'foot.dae',
)
_MIN_MESHED_USD_BYTES = 100_000


def build_fallback_go2_urdf() -> str:
    """Return a mesh-free Go2 URDF with the official kinematic dimensions."""

    robot = ET.Element('robot', {'name': 'Go2'})
    ET.SubElement(robot, 'material', {'name': 'go2_dark'}).append(
        ET.Element('color', {'rgba': '0.12 0.14 0.16 1'})
    )
    ET.SubElement(robot, 'material', {'name': 'go2_orange'}).append(
        ET.Element('color', {'rgba': '0.95 0.42 0.08 1'})
    )
    ET.SubElement(robot, 'material', {'name': 'go2_light'}).append(
        ET.Element('color', {'rgba': '0.72 0.75 0.78 1'})
    )

    base = ET.SubElement(robot, 'link', {'name': 'base'})
    _add_inertial(
        base,
        mass=6.921,
        origin=(0.021112, 0.0, -0.005366),
        inertia={
            'ixx': 0.02448,
            'ixy': 0.00012166,
            'ixz': 0.0014849,
            'iyy': 0.098077,
            'iyz': -0.0000312,
            'izz': 0.107,
        },
    )
    _add_shape(base, 'visual', 'box', (0.42, 0.16, 0.13), material='go2_dark')
    _add_shape(base, 'collision', 'box', (0.3762, 0.0935, 0.114))

    _add_head(robot)
    for prefix, front_sign, side_sign in (
        ('FL', 1.0, 1.0),
        ('FR', 1.0, -1.0),
        ('RL', -1.0, 1.0),
        ('RR', -1.0, -1.0),
    ):
        _add_leg(robot, prefix, front_sign, side_sign)

    ET.indent(robot, space='  ')
    return ET.tostring(robot, encoding='unicode', xml_declaration=True)


def official_go2_meshes_ready(asset_dir: str | Path) -> bool:
    """Return True when the official Unitree visual meshes are on disk."""

    mesh_dir = Path(asset_dir).expanduser() / 'dae'
    return all(
        (mesh_dir / name).is_file() and (mesh_dir / name).stat().st_size > 1000
        for name in OFFICIAL_GO2_MESH_NAMES
    )


def prepare_official_go2_urdf(asset_dir: str | Path) -> Path:
    """Rewrite package:// mesh URIs to local files for the Isaac URDF importer."""

    root = Path(asset_dir).expanduser().resolve()
    source = root / 'go2_description.urdf'
    if not source.is_file():
        raise FileNotFoundError(f'official Go2 URDF not found: {source}')
    mesh_dir = (root / 'dae').resolve()
    rewritten = source.read_text(encoding='utf-8').replace(
        'package://go2_description/dae/',
        f'{mesh_dir}/',
    )
    output = root / 'go2_official.urdf'
    output.write_text(rewritten, encoding='utf-8')
    return output


def is_isaaclab_go2_usd(path: str | Path) -> bool:
    """Return True when the file looks like NVIDIA's packed Go2 USDC."""

    usd_path = Path(path).expanduser()
    if not usd_path.is_file() or usd_path.stat().st_size < _MIN_ISAACLAB_USD_BYTES:
        return False
    with usd_path.open('rb') as handle:
        return handle.read(8) == b'PXR-USDC'


def is_composed_go2_usd(path: str | Path) -> bool:
    """Recognize a usable referenced Go2 asset without a file-size heuristic.

    A small root layer can contain the complete rigid-body topology while its
    collision meshes live in referenced layers. This is structural validation,
    not a claim that an arbitrary file is an official model.
    """
    path=Path(path).expanduser()
    if not path.is_file():return False
    try:
        from pxr import Usd,UsdGeom,UsdPhysics
        stage=Usd.Stage.Open(str(path))
        if stage is None or not stage.GetDefaultPrim():return False
        joints={p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
        expected={f'{leg}_{part}_joint' for leg in ['FL','FR','RL','RR'] for part in ['hip','thigh','calf']}
        base=any(p.GetName()=='base' and p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
        collision=any(p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Gprim)
            for p in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()))
        return base and expected.issubset(joints) and collision
    except Exception:
        return False


def ensure_go2_usd(usd_path: str, generate_fallback: bool = True) -> str:
    """Return a usable Go2 USD path, preferring the Isaac Lab official asset."""

    if '://' in usd_path:
        return usd_path
    output_path = Path(usd_path).expanduser().resolve()
    if is_isaaclab_go2_usd(output_path) or is_composed_go2_usd(output_path):
        return str(output_path)

    sibling_isaaclab = output_path.parent / 'isaaclab_go2.usd'
    if is_isaaclab_go2_usd(sibling_isaaclab):
        return str(sibling_isaaclab)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _try_download_official_usd(output_path):
        return str(output_path)
    if output_path.name != sibling_isaaclab.name and _try_download_official_usd(
        sibling_isaaclab
    ):
        return str(sibling_isaaclab)

    if official_go2_meshes_ready(output_path.parent):
        official_urdf = prepare_official_go2_urdf(output_path.parent)
        _import_urdf_to_usd(official_urdf, output_path)
        if _is_meshed_usd(output_path):
            return str(output_path)

    if output_path.is_file() and generate_fallback:
        return str(output_path)
    if not generate_fallback:
        raise FileNotFoundError(f'Go2 USD not found: {output_path}')

    fallback_urdf = output_path.with_suffix('.urdf')
    fallback_urdf.write_text(build_fallback_go2_urdf(), encoding='utf-8')
    _import_urdf_to_usd(fallback_urdf, output_path)
    if not output_path.is_file():
        raise RuntimeError(f'failed to generate fallback Go2 USD at {output_path}')
    return str(output_path)


def _is_meshed_usd(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= _MIN_MESHED_USD_BYTES


def _try_download_official_usd(output_path: Path) -> bool:
    try:
        import omni.client
    except ImportError:
        return False

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeoutError

    def _copy(url: str):
        return omni.client.copy(url, str(output_path))

    ok = getattr(omni.client.Result, 'OK', 0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        for url in OFFICIAL_GO2_USD_CANDIDATES:
            try:
                result = pool.submit(_copy, url).result(timeout=12)
            except (Exception, FutureTimeoutError):
                continue
            if result == ok and _is_meshed_usd(output_path):
                return True
    return False


def _import_urdf_to_usd(urdf_path: Path, output_path: Path):
    from omni.isaac.core.utils.extensions import enable_extension

    enable_extension('omni.importer.urdf')
    import omni.kit.commands

    status, import_config = omni.kit.commands.execute('URDFCreateImportConfig')
    if not status:
        raise RuntimeError('failed to create the Isaac Sim URDF import configuration')
    import_config.fix_base = False
    import_config.import_inertia_tensor = True
    import_config.make_default_prim = True
    import_config.merge_fixed_joints = False
    import_config.self_collision = False
    imported, _ = omni.kit.commands.execute(
        'URDFParseAndImportFile',
        urdf_path=str(urdf_path),
        import_config=import_config,
        dest_path=str(output_path),
    )
    if not imported or not output_path.is_file():
        raise RuntimeError(f'failed to import Go2 URDF into USD at {output_path}')


def _add_head(robot: ET.Element):
    upper = ET.SubElement(robot, 'link', {'name': 'Head_upper'})
    _add_inertial(upper, 0.001, (0.0, 0.0, 0.0), _small_inertia())
    _add_shape(
        upper,
        'visual',
        'cylinder',
        (0.05, 0.09),
        material='go2_orange',
    )
    _add_shape(upper, 'collision', 'cylinder', (0.05, 0.09))
    _add_fixed_joint(
        robot,
        'Head_upper_joint',
        parent='base',
        child='Head_upper',
        origin=(0.285, 0.0, 0.01),
        preserve=True,
    )

    lower = ET.SubElement(robot, 'link', {'name': 'Head_lower'})
    _add_inertial(lower, 0.001, (0.0, 0.0, 0.0), _small_inertia())
    _add_shape(
        lower,
        'visual',
        'sphere',
        (0.047,),
        material='go2_dark',
    )
    _add_shape(lower, 'collision', 'sphere', (0.047,))
    _add_fixed_joint(
        robot,
        'Head_lower_joint',
        parent='Head_upper',
        child='Head_lower',
        origin=(0.008, 0.0, -0.07),
        preserve=True,
    )


def _add_leg(
    robot: ET.Element,
    prefix: str,
    front_sign: float,
    side_sign: float,
):
    hip = ET.SubElement(robot, 'link', {'name': f'{prefix}_hip'})
    _add_inertial(
        hip,
        0.678,
        (
            -front_sign * 0.0054,
            side_sign * 0.00194,
            -0.000105,
        ),
        {
            'ixx': 0.00048,
            'ixy': -front_sign * side_sign * 0.00000301,
            'ixz': front_sign * 0.00000111,
            'iyy': 0.000884,
            'iyz': -side_sign * 0.00000142,
            'izz': 0.000596,
        },
    )
    hip_shape_origin = (0.0, side_sign * 0.08, 0.0)
    hip_shape_rpy = (math.pi / 2.0, 0.0, 0.0)
    _add_shape(
        hip,
        'visual',
        'cylinder',
        (0.046, 0.04),
        origin=hip_shape_origin,
        rpy=hip_shape_rpy,
        material='go2_light',
    )
    _add_shape(
        hip,
        'collision',
        'cylinder',
        (0.046, 0.04),
        origin=hip_shape_origin,
        rpy=hip_shape_rpy,
    )
    _add_revolute_joint(
        robot,
        f'{prefix}_hip_joint',
        parent='base',
        child=f'{prefix}_hip',
        origin=(front_sign * 0.1934, side_sign * 0.0465, 0.0),
        axis=(1.0, 0.0, 0.0),
        limits=(-1.0472, 1.0472, 23.7, 30.1),
    )

    thigh = ET.SubElement(robot, 'link', {'name': f'{prefix}_thigh'})
    _add_inertial(
        thigh,
        1.152,
        (-0.00374, -side_sign * 0.0223, -0.0327),
        {
            'ixx': 0.00584,
            'ixy': side_sign * 0.0000872,
            'ixz': -0.000289,
            'iyy': 0.0058,
            'iyz': side_sign * 0.000808,
            'izz': 0.00103,
        },
    )
    _add_shape(
        thigh,
        'visual',
        'box',
        (0.055, 0.05, 0.213),
        origin=(0.0, 0.0, -0.1065),
        material='go2_orange',
    )
    _add_shape(
        thigh,
        'collision',
        'box',
        (0.034, 0.0245, 0.213),
        origin=(0.0, 0.0, -0.1065),
    )
    thigh_limits = (
        (-1.5708, 3.4907, 23.7, 30.1)
        if front_sign > 0
        else (-0.5236, 4.5379, 23.7, 30.1)
    )
    _add_revolute_joint(
        robot,
        f'{prefix}_thigh_joint',
        parent=f'{prefix}_hip',
        child=f'{prefix}_thigh',
        origin=(0.0, side_sign * 0.0955, 0.0),
        axis=(0.0, 1.0, 0.0),
        limits=thigh_limits,
    )

    calf = ET.SubElement(robot, 'link', {'name': f'{prefix}_calf'})
    _add_inertial(
        calf,
        0.154,
        (0.00548, -side_sign * 0.000975, -0.115),
        {
            'ixx': 0.00108,
            'ixy': side_sign * 0.00000034,
            'ixz': 0.0000172,
            'iyy': 0.0011,
            'iyz': side_sign * 0.00000828,
            'izz': 0.0000329,
        },
    )
    _add_shape(
        calf,
        'visual',
        'cylinder',
        (0.018, 0.12),
        origin=(0.008, 0.0, -0.06),
        rpy=(0.0, -0.21, 0.0),
        material='go2_dark',
    )
    _add_shape(
        calf,
        'collision',
        'cylinder',
        (0.012, 0.12),
        origin=(0.008, 0.0, -0.06),
        rpy=(0.0, -0.21, 0.0),
    )
    _add_revolute_joint(
        robot,
        f'{prefix}_calf_joint',
        parent=f'{prefix}_thigh',
        child=f'{prefix}_calf',
        origin=(0.0, 0.0, -0.213),
        axis=(0.0, 1.0, 0.0),
        limits=(-2.7227, -0.83776, 45.43, 15.70),
    )

    foot = ET.SubElement(robot, 'link', {'name': f'{prefix}_foot'})
    _add_inertial(foot, 0.04, (0.0, 0.0, 0.0), _small_inertia())
    _add_shape(
        foot,
        'visual',
        'sphere',
        (0.022,),
        origin=(-0.002, 0.0, 0.0),
        material='go2_light',
    )
    _add_shape(
        foot,
        'collision',
        'sphere',
        (0.022,),
        origin=(-0.002, 0.0, 0.0),
    )
    _add_fixed_joint(
        robot,
        f'{prefix}_foot_joint',
        parent=f'{prefix}_calf',
        child=f'{prefix}_foot',
        origin=(0.0, 0.0, -0.213),
        preserve=True,
    )


def _add_inertial(
    link: ET.Element,
    mass: float,
    origin: Sequence[float],
    inertia: Mapping[str, float],
):
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(
        inertial,
        'origin',
        {'xyz': _vector(origin), 'rpy': '0 0 0'},
    )
    ET.SubElement(inertial, 'mass', {'value': _number(mass)})
    ET.SubElement(
        inertial,
        'inertia',
        {key: _number(value) for key, value in inertia.items()},
    )


def _add_shape(
    link: ET.Element,
    kind: str,
    shape: str,
    dimensions: Sequence[float],
    origin=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
    material: str = '',
):
    node = ET.SubElement(link, kind)
    ET.SubElement(
        node,
        'origin',
        {'xyz': _vector(origin), 'rpy': _vector(rpy)},
    )
    geometry = ET.SubElement(node, 'geometry')
    if shape == 'box':
        ET.SubElement(geometry, 'box', {'size': _vector(dimensions)})
    elif shape == 'cylinder':
        ET.SubElement(
            geometry,
            'cylinder',
            {
                'radius': _number(dimensions[0]),
                'length': _number(dimensions[1]),
            },
        )
    elif shape == 'sphere':
        ET.SubElement(geometry, 'sphere', {'radius': _number(dimensions[0])})
    else:
        raise ValueError(f'unsupported URDF shape: {shape}')
    if kind == 'visual' and material:
        ET.SubElement(node, 'material', {'name': material})


def _add_revolute_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    origin: Sequence[float],
    axis: Sequence[float],
    limits: Sequence[float],
):
    joint = ET.SubElement(robot, 'joint', {'name': name, 'type': 'revolute'})
    ET.SubElement(joint, 'origin', {'xyz': _vector(origin), 'rpy': '0 0 0'})
    ET.SubElement(joint, 'parent', {'link': parent})
    ET.SubElement(joint, 'child', {'link': child})
    ET.SubElement(joint, 'axis', {'xyz': _vector(axis)})
    ET.SubElement(
        joint,
        'limit',
        {
            'lower': _number(limits[0]),
            'upper': _number(limits[1]),
            'effort': _number(limits[2]),
            'velocity': _number(limits[3]),
        },
    )
    ET.SubElement(joint, 'dynamics', {'damping': '0', 'friction': '0'})


def _add_fixed_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    origin: Sequence[float],
    preserve: bool = False,
):
    attributes = {'name': name, 'type': 'fixed'}
    if preserve:
        attributes['dont_collapse'] = 'true'
    joint = ET.SubElement(robot, 'joint', attributes)
    ET.SubElement(joint, 'origin', {'xyz': _vector(origin), 'rpy': '0 0 0'})
    ET.SubElement(joint, 'parent', {'link': parent})
    ET.SubElement(joint, 'child', {'link': child})


def _small_inertia():
    return {
        'ixx': 0.0000096,
        'ixy': 0.0,
        'ixz': 0.0,
        'iyy': 0.0000096,
        'iyz': 0.0,
        'izz': 0.0000096,
    }


def _vector(values: Sequence[float]) -> str:
    return ' '.join(_number(value) for value in values)


def _number(value: float) -> str:
    return f'{float(value):.12g}'
