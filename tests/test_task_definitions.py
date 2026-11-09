import math
import pytest
from grutopia_extension.interactive_navigation.task_definitions import online_task_definition,task_initial_heading


@pytest.mark.parametrize('task,expected',[('T01',0.),('T02',math.pi),('T03',math.pi/2),('T04',0.),('T05',math.pi)])
def test_existing_tasks_keep_recorded_start_orientation(task,expected):
    assert task_initial_heading({'id':task}) == expected


def test_new_task_heading_survives_without_offline_geometry():
    task={'id':'T06','start':[1.,2.],'goal':[3.,4.],'initial_heading_rad':-math.pi/2,
          'shortest_geometry':{'path':[[1.,2.],[3.,4.]]},'decision':'push','box_id':'B06'}
    online=online_task_definition(task)
    assert online == {'id':'T06','start':[1.,2.],'goal':[3.,4.],'initial_heading_rad':-math.pi/2}
    assert task_initial_heading(online) == -math.pi/2


def test_new_task_without_declared_heading_rejected():
    with pytest.raises(ValueError,match='requires initial_heading_rad'):
        online_task_definition({'id':'T10','start':[1.,2.],'goal':[3.,4.]})


@pytest.mark.parametrize('heading',[float('nan'),float('inf'),-float('inf')])
def test_invalid_start_heading_rejected(heading):
    with pytest.raises(ValueError,match='finite'):
        task_initial_heading({'id':'T06','initial_heading_rad':heading})
