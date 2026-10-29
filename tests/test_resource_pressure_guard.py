from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from go2_resource_pressure import MemoryPressureGuard


def test_short_memory_dip_does_not_suspend_and_timer_resets():
    guard=MemoryPressureGuard()
    assert guard.update(0,15) is None
    assert guard.update(9,15) is None
    assert guard.update(10,17) is None
    assert guard.update(11,15) is None
    assert guard.update(20,15) is None
    assert guard.update(21,15)['action']=='pause'
    assert guard.update(22,15) is None


def test_pause_holds_through_pressure_then_requires_sustained_recovery():
    guard=MemoryPressureGuard();guard.update(0,15);assert guard.update(10,15)['action']=='pause'
    assert guard.update(70,10) is None
    assert guard.update(80,56) is None
    assert guard.update(109,56) is None
    assert guard.update(110,55.9) is None
    assert guard.update(111,60) is None
    assert guard.update(141,60)['action']=='resume'
    assert guard.report()['pause_count']==guard.report()['resume_count']==1
    assert guard.update(142,60) is None


@pytest.mark.parametrize('paused',[False,True])
def test_emergency_immediately_kills_running_or_suspended_job(paused):
    guard=MemoryPressureGuard()
    if paused:guard.update(0,15);guard.update(10,15)
    assert guard.update(11,3.9)=={'action':'kill','reason':'host_memory_emergency'}
    assert guard.update(12,3.9) is None


def test_pause_timeout_requests_orderly_abort_then_has_bounded_grace():
    guard=MemoryPressureGuard();guard.update(0,15);guard.update(10,15)
    assert guard.update(609,20) is None
    assert guard.update(610,20)=={'action':'abort','reason':'resource_pause_timeout'}
    assert guard.update(909,60) is None
    assert guard.update(910,60)['action']=='kill'


def test_recovery_at_pause_deadline_can_complete_its_stability_window():
    guard=MemoryPressureGuard();guard.update(0,15);guard.update(10,15)
    assert guard.update(609,56) is None
    assert guard.update(610,56) is None
    assert guard.update(639,56)['action']=='resume'


def test_invalid_samples_cannot_silently_make_resource_decisions():
    guard=MemoryPressureGuard();guard.update(10,60)
    with pytest.raises(ValueError):guard.update(9,60)
    with pytest.raises(ValueError):guard.update(11,float('nan'))
    with pytest.raises(ValueError):guard.update(11,-1)
