from types import SimpleNamespace

import pytest

from grutopia_extension.interactive_navigation.go2_physics_backend import (
    collect_go2_physics_contract,
    configure_go2_physics_backend,
)


def context(gpu=True, broadphase='GPU'):
    prim = SimpleNamespace(GetPath=lambda: '/physicsScene', GetAttributes=lambda: [])
    # Isaac Core 4.2 retains these CPU flags when GPU dynamics uses CPU readback.
    return SimpleNamespace(
        _physics_scene=SimpleNamespace(GetPrim=lambda: prim),
        device='cpu', use_gpu_sim=False, use_gpu_pipeline=False,
        is_gpu_dynamics_enabled=lambda: gpu,
        get_broadphase_type=lambda: broadphase,
        get_solver_type=lambda: 'TGS',
    )


def test_cpu_readback_does_not_falsely_reject_gpu_scene():
    report = collect_go2_physics_contract(context(), 'gpu_dynamics')
    assert report['gpu_dynamics'] and report['broadphase'] == 'GPU'
    assert report['device'] == 'cpu' and not report['use_gpu_pipeline']


@pytest.mark.parametrize('gpu,broadphase', [(False, 'GPU'), (True, 'MBP')])
def test_initialization_cannot_silently_override_requested_backend(gpu, broadphase):
    with pytest.raises(RuntimeError, match='did not survive'):
        collect_go2_physics_contract(context(gpu, broadphase), 'gpu_dynamics')


def test_default_keeps_historical_world_configuration():
    # Default accepts a context with no setters and reports its actual settings.
    cpu = context(False, 'MBP')
    configure_go2_physics_backend(cpu, 'default')
    assert collect_go2_physics_contract(cpu, 'default')['broadphase'] == 'MBP'


def test_unknown_backend_is_rejected_before_using_context():
    with pytest.raises(ValueError, match='Unsupported'):
        configure_go2_physics_backend(None, 'unrecognized')
