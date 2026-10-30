"""Explicit PhysX selection, with CPU readback retained for the Go2 adapter."""

PHYSICS_BACKENDS = ('default', 'gpu_dynamics')


def configure_go2_physics_backend(context, backend: str) -> None:
    """Configure before World.reset; do not change the tensor data pipeline."""
    if backend not in PHYSICS_BACKENDS:
        raise ValueError(f'Unsupported Go2 physics backend: {backend}')
    if backend == 'gpu_dynamics':
        context.enable_gpu_dynamics(True)
        context.set_broadphase_type('GPU')
        context.enable_ccd(False)


def collect_go2_physics_contract(context, backend: str) -> dict:
    """Check the scene after reset, rather than trusting cached device flags."""
    if backend not in PHYSICS_BACKENDS:
        raise ValueError(f'Unsupported Go2 physics backend: {backend}')
    prim = context._physics_scene.GetPrim()
    report = {
        'requested_backend': backend,
        'scene_path': str(prim.GetPath()),
        'device': context.device,
        'cached_use_gpu_sim': bool(context.use_gpu_sim),
        'use_gpu_pipeline': bool(context.use_gpu_pipeline),
        'gpu_dynamics': bool(context.is_gpu_dynamics_enabled()),
        'broadphase': context.get_broadphase_type(),
        'solver': context.get_solver_type(),
        'scene_attributes': {a.GetName(): str(a.Get()) for a in prim.GetAttributes()},
    }
    if backend == 'gpu_dynamics' and (
        not report['gpu_dynamics'] or report['broadphase'] != 'GPU'
    ):
        raise RuntimeError('Requested Go2 GPU dynamics/broadphase did not survive scene initialization')
    return report
