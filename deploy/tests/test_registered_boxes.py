"""Registered LiDAR geometry and identity contracts; no ROS, SDK or simulator."""
from copy import deepcopy
import math
import threading
import time

import numpy as np
import pytest

from deploy.perception.box_fitting import fit_upright_box, scan_candidates
from deploy.perception.registered_boxes import (
    DEFAULT_TRACKING, RegisteredBoxPerception, validate_tracking_config,
)
from deploy.robot.observation import Snapshot


DIMS = (.4, .3, .5)


def cloud(center=(1., .5), yaw=.3, noise=.002, seed=2, single=False, dims=DIMS):
    u, z = np.meshgrid(np.linspace(-1, 1, 18), np.linspace(.03, dims[2]-.005, 14))
    a = np.c_[np.full(u.size, -dims[0]/2), u.ravel()*dims[1]/2, z.ravel()]
    b = np.c_[u.ravel()*dims[0]/2, np.full(u.size, -dims[1]/2), z.ravel()]
    p = a if single else np.vstack([a, b])
    c, s = np.cos(yaw), np.sin(yaw)
    p[:, :2] = p[:, :2] @ np.array([[c, s], [-s, c]]) + center
    return p + np.random.default_rng(seed).normal(0., noise, p.shape)


def origin_for(center=(1., .5), yaw=.3):
    c, s = np.cos(yaw), np.sin(yaw)
    xy = np.array([-1.5, -1.2]) @ np.array([[c, s], [-s, c]]) + center
    return np.r_[xy, .3]


def config(**options):
    return {'geometry_verified': True, 'registry_reference': 'test-measured-registry',
            'boxes': [{'id': 'B1', 'dimensions_m': list(DIMS),
                       'search_region': [.5, 1.5, 0., 1.], 'push_allowed': True}], **options}


def packet(index=1, stamp=0., points=None, *, center=(1., .5), yaw=.3):
    p = cloud(center, yaw) if points is None else points
    return Snapshot({'position': np.array([0., -1., .3]),
                     'sensors': {'lidar': {'pointcloud': p, 'position': origin_for(center, yaw),
                                            'physics_step': index}}},
                    stamp, stamp, stamp, None, (0., 0., 0.), index,
                    planning_frame='experiment', coordinate_transform={'source_frame': 'odom'})


def engine(**options):
    return RegisteredBoxPerception(config(**options), synchronous=True)


def establish(provider, center=(1., .5), yaw=.3):
    assert provider.update(packet(1, 0., center=center, yaw=yaw), 0.)[0] == ()
    boxes, store = provider.update(packet(2, .1, center=center, yaw=yaw), .1)
    assert store is provider and len(boxes) == 1
    return boxes[0]


@pytest.mark.parametrize('yaw', [-1.1, -.6, -.1, 0., .3, .9, 1.4])
def test_known_size_fit_recovers_measured_translation_rotation_and_full_geometry(yaw):
    fit = fit_upright_box(cloud(yaw=yaw), DIMS, origin_for(yaw=yaw), DEFAULT_TRACKING)
    assert fit is not None
    assert fit.center == pytest.approx((1., .5, .25), abs=.008)
    assert fit.yaw_rad == pytest.approx(yaw, abs=.025)
    assert fit.dimensions == DIMS
    assert len(fit.visible_faces) == 2 and fit.residual_m < .01


def test_empty_single_plane_and_truncated_boundaries_are_not_box_measurements():
    assert scan_candidates(np.empty((0, 3)), DIMS, origin_for(), DEFAULT_TRACKING) == ()
    assert fit_upright_box(cloud(single=True), DIMS, origin_for(), DEFAULT_TRACKING) is None
    p = cloud(yaw=0., noise=0.)
    truncated = p[p[:, 0] < 1.04]
    assert fit_upright_box(truncated, DIMS, origin_for(yaw=0.), DEFAULT_TRACKING) is None


def test_wall_attached_to_box_is_not_cropped_into_registered_roi():
    p = cloud(yaw=0., noise=0.)
    u, z = np.meshgrid(np.linspace(1.2, 2.5, 80), np.linspace(.03, .49, 14))
    wall = np.c_[u.ravel(), np.full(u.size, .35), z.ravel()]
    assert scan_candidates(np.vstack([p, wall]), DIMS, origin_for(yaw=0.), DEFAULT_TRACKING) == ()


def test_unconnected_close_neighbour_invalidates_fit():
    p = cloud(yaw=0., noise=0.)
    u, z = np.meshgrid(np.linspace(.85, 1.15, 10), np.linspace(.03, .49, 10))
    neighbour = np.c_[u.ravel(), np.full(u.size, .72), z.ravel()]
    assert scan_candidates(np.vstack([p, neighbour]), DIMS, origin_for(yaw=0.), DEFAULT_TRACKING) == ()


def test_floor_and_top_alone_do_not_supply_vertical_box_pose():
    p = cloud(noise=0.)
    p[:, 2] = DIMS[2]
    assert fit_upright_box(p, DIMS, origin_for(), DEFAULT_TRACKING) is None
    raised = cloud(noise=0.) + [0, 0, .15]
    assert fit_upright_box(raised, DIMS, origin_for(), DEFAULT_TRACKING) is None


def test_registration_permission_is_separate_and_has_actual_lidar_evidence():
    provider = engine()
    try:
        box = establish(provider)
        permission = provider.permission('B1', .11)
        assert permission.source == 'registered_box_lidar'
        assert permission.classifier_models == ()
        assert permission.evidence_frames == ('lidar:0:1', 'lidar:0:2')
        assert len(permission.source_digest) == 64
        assert permission.source_reference == 'test-measured-registry'
        assert provider.expires_at('B1') == pytest.approx(.35)
        assert provider.eligible_ids(.11) == {'B1'}
        assert provider.permission('B1', .35) is None
        assert provider.permission('unknown', .11) is None
        assert box.yaw == 0.  # AABB must not be rotated a second time.
        assert box.local_dimensions == DIMS and box.mass_kg is None
        expected_width = math.cos(.3)*.4+math.sin(.3)*.3+.06
        assert box.dimensions[0] == pytest.approx(expected_width, abs=.008)
        assert box.orientation[3] == pytest.approx(math.sin(.3/2), abs=.015)
    finally:
        provider.close()


def test_moving_rotating_box_follows_measurement_beyond_initial_search_region():
    cfg = config()
    cfg['boxes'][0]['search_region'] = [.97, 1.03, .47, .53]
    provider = RegisteredBoxPerception(cfg, synchronous=True)
    try:
        establish(provider)
        for i in range(3, 9):
            center = (1.+.035*(i-2), .5)
            yaw = .3+.025*(i-2)
            boxes, _ = provider.update(packet(i, (i-1)*.1, center=center, yaw=yaw), (i-1)*.1)
            assert len(boxes) == 1 and boxes[0].xy == pytest.approx(center, abs=.01)
            assert provider.permission('B1', (i-1)*.1) is not None
    finally:
        provider.close()


def test_repeat_scan_does_not_confirm_or_renew_and_nonincreasing_scan_revokes():
    provider = engine()
    try:
        p = packet()
        provider.update(p, 0.)
        provider.update(p, .1)
        assert provider.snapshot(.1)['B1']['frames'] == 1
        boxes, _ = provider.update(packet(2, .11), .11)
        assert len(boxes) == 1
        assert provider.expires_at('B1') == pytest.approx(.36)
        assert provider.update(packet(1, .12), .12)[0] == ()
        assert provider.permission('B1', .12) is None
        assert 'Non-increasing' in provider.last_error
    finally:
        provider.close()


def test_no_geometry_acceptance_still_reports_measured_fits_without_permission():
    provider = engine(geometry_verified=False)
    try:
        provider.update(packet(), 0.)
        assert provider.update(packet(2, .1), .1)[0] == ()
        assert provider.permission('B1', .1) is None
        detection = provider.last_delivery['detections'][0]
        assert detection['center'] == pytest.approx((1., .5, .25), abs=.01)
        assert detection['geometry_verified'] is False
        assert detection['dimensions_source'] == 'measured_registry'
    finally:
        provider.close()


def test_unpermitted_registry_box_remains_geometry_but_cannot_be_pushed():
    cfg = config()
    cfg['boxes'][0]['push_allowed'] = False
    provider = RegisteredBoxPerception(cfg, synchronous=True)
    try:
        establish(provider)
        assert provider.eligible_ids(.1) == frozenset()
    finally:
        provider.close()


def test_initial_multiple_candidates_and_registry_id_collision_reject_permissions():
    provider = engine()
    try:
        two = np.vstack([cloud(center=(.65, .25)), cloud(center=(1.35, .75))])
        provider.update(packet(points=two), 0.)
        assert provider.update(packet(2, .1, two), .1)[0] == ()
        assert provider.permission('B1', .1) is None
    finally:
        provider.close()
    cfg = config()
    cfg['boxes'].append({**deepcopy(cfg['boxes'][0]), 'id': 'B2'})
    provider = RegisteredBoxPerception(cfg, synchronous=True)
    try:
        provider.update(packet(), 0.)
        assert provider.update(packet(2, .1), .1)[0] == ()
        assert 'multiple registered identities' in provider.last_delivery['detections'][0]['invalid_reason']
    finally:
        provider.close()


def test_lost_identity_does_not_transfer_permission_and_reset_rebuilds_from_scan():
    provider = engine()
    try:
        establish(provider)
        assert provider.update(packet(3, .2, np.empty((0, 3))), .2)[0] == ()
        assert provider.update(packet(4, .3), .3)[0] == ()
        provider.reset()
        assert provider.update(packet(5, .4), .4)[0] == ()
        assert len(provider.update(packet(6, .5), .5)[0]) == 1
        assert provider.permission('B1', .5).evidence_frames == ('lidar:1:5', 'lidar:1:6')
    finally:
        provider.close()


def test_pose_jump_and_expired_geometry_require_new_identity_confirmation():
    provider = engine()
    try:
        establish(provider)
        assert provider.update(packet(3, .2, center=(1.4, .5)), .2)[0] == ()
        assert provider.permission('B1', .2) is None
        provider.reset()
        establish(provider)
        assert provider.update(packet(3, .4), .4)[0] == ()
        assert provider.permission('B1', .4) is None
    finally:
        provider.close()


@pytest.mark.parametrize('kind', ['future', 'stale', 'uncalibrated', 'frame', 'transform', 'nan', 'pose'])
def test_invalid_frame_revokes_current_geometry(kind):
    provider = engine()
    try:
        establish(provider)
        p = packet(3, .2)
        now = .2
        if kind == 'future': p.lidar_at = .21
        if kind == 'stale': p.lidar_at = -.1
        if kind == 'uncalibrated': p.geometry_calibrated = False
        if kind == 'frame': p.planning_frame = 'odom'
        if kind == 'transform': p.coordinate_transform = {'source_frame': 'new_odom'}
        if kind == 'nan': p.observation['sensors']['lidar']['pointcloud'][0, 0] = np.nan
        if kind == 'pose': p.pose_at = -.1
        assert provider.update(p, now)[0] == ()
        assert provider.permission('B1', now) is None
    finally:
        provider.close()


def wait_for(condition):
    deadline = time.monotonic()+2.
    while not condition() and time.monotonic() < deadline:
        time.sleep(.002)
    assert condition()


def test_async_worker_is_bounded_nonblocking_and_drops_late_result():
    provider = RegisteredBoxPerception(config())
    entered, release = threading.Event(), threading.Event()
    original = provider._fit
    def blocked(frame):
        entered.set()
        release.wait(2.)
        return original(frame)
    provider._fit = blocked
    try:
        start = time.monotonic()
        provider.update(packet(), 0.)
        assert time.monotonic()-start < .1
        wait_for(entered.is_set)
        for i in range(2, 8):
            provider.update(packet(i, i*.01), i*.01)
        assert provider.jobs.qsize() == 1
        assert provider.jobs.queue[0].frame_id == 7
        release.set()
        wait_for(lambda: not provider.results.empty() and provider.results.queue[0][0].frame_id == 7)
        provider.update(packet(8, .5), .5)
        assert provider.permission('B1', .5) is None
        assert 'stale' in provider.last_error
    finally:
        release.set()
        provider.close()


def test_reset_during_inference_never_delivers_old_generation():
    provider = RegisteredBoxPerception(config())
    entered, release = threading.Event(), threading.Event()
    original = provider._fit
    def blocked(frame):
        entered.set()
        release.wait(2.)
        return original(frame)
    provider._fit = blocked
    try:
        provider.update(packet(), 0.)
        wait_for(entered.is_set)
        provider.reset()
        release.set()
        time.sleep(.03)
        assert provider.results.empty() and provider.last_delivery is None
        provider._fit = original
        provider.update(packet(2, .1), .1)
        wait_for(lambda: not provider.results.empty())
        assert provider.update(packet(2, .1), .1)[0] == ()
        assert provider.last_delivery['session_id'] == 1
    finally:
        release.set()
        provider.close()


@pytest.mark.parametrize('edit', [
    {'max_age_s': True}, {'max_age_s': float('nan')}, {'max_age_s': 1.},
    {'min_points': 20.5}, {'min_face_coverage': .1}, {'geometry_verified': 'yes'},
    {'registry_reference': ''}, {'unknown': 1}, {'boxes': []},
])
def test_config_strictly_rejects_invalid_values(edit):
    with pytest.raises(ValueError):
        validate_tracking_config(config(**edit))


@pytest.mark.parametrize('field,value', [('dimensions_m', [.4, True, .5]),
    ('dimensions_m', [.4, float('inf'), .5]), ('search_region', [1., 0., 0., 1.]),
    ('push_allowed', 1), ('id', '')])
def test_registry_validation(field, value):
    cfg = config()
    cfg['boxes'][0][field] = value
    with pytest.raises(ValueError):
        validate_tracking_config(cfg)


def test_square_pose_symmetry_does_not_create_identity_or_yaw_jump():
    cfg = config()
    cfg['boxes'][0]['dimensions_m'] = [.4, .4, .5]
    provider = RegisteredBoxPerception(cfg, synchronous=True)
    try:
        previous_yaw = None
        for i, yaw in enumerate([.72, .77, .82, .87, .92], start=1):
            points = cloud(yaw=yaw, dims=(.4, .4, .5))
            provider.update(packet(i, i*.1, points, yaw=yaw), i*.1)
            measured = provider.last_delivery['detections'][0]['yaw_rad']
            if previous_yaw is not None:
                assert abs(measured-previous_yaw) < .10
                assert provider.permission('B1', i*.1) is not None
            previous_yaw = measured
    finally:
        provider.close()


def test_tall_pillar_cannot_become_box_by_height_crop():
    short = cloud(yaw=0., noise=0.)
    tall = np.vstack([short, short+[0., 0., .5]])
    assert scan_candidates(tall, DIMS, origin_for(yaw=0.), DEFAULT_TRACKING) == ()


def test_tilted_box_rejected_by_upright_surface_model():
    p = cloud(yaw=0., noise=0.)
    local = p-[1., .5, .25]
    angle = math.radians(25.)
    c, s = math.cos(angle), math.sin(angle)
    tilted = local@np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T+[1., .5, .25]
    assert fit_upright_box(tilted, DIMS, origin_for(yaw=0.), DEFAULT_TRACKING) is None


def bounded_face_cloud(center=(1., .5), yaw=.3, *, depth_to_face=.45, sides=(-1, 1),
                       truncated=False, wall=False, noise=.001):
    """One observed -X face plus measured rays passing OUTSIDE both edges."""
    width, height = DIMS[1], DIMS[2]
    yy, zz = np.meshgrid(np.linspace(-width/2, width/2, 22), np.linspace(.03, height-.005, 16))
    face = np.c_[np.full(yy.size, -DIMS[0]/2), yy.ravel(), zz.ravel()]
    if truncated:
        face = face[face[:, 1] > -.07]
    origin = np.array([-DIMS[0]/2-depth_to_face, 0., .3])
    background = []
    for sign in sides:
        for z in (.25, .29, .33, .37, .41):
            crossing = np.array([-DIMS[0]/2, sign*(width/2+.04), z])
            scale = (DIMS[0]/2+.25-origin[0])/depth_to_face
            background.append(crossing if wall else origin+scale*(crossing-origin))
    points = np.vstack([face, np.asarray(background).reshape(-1, 3)])
    points += np.random.default_rng(8).normal(0., noise, points.shape)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, s], [-s, c]])
    points[:, :2] = points[:, :2]@rotation+center
    origin[:2] = origin[:2]@rotation+center
    return points, origin


def front_packet(index, stamp, **kwargs):
    points, origin = bounded_face_cloud(**kwargs)
    result = packet(index, stamp, points)
    result.observation['sensors']['lidar']['position'] = origin
    return result


def test_confirmed_track_uses_new_bounded_face_position_through_centered_contact():
    provider = engine()
    try:
        establish(provider)
        for index in range(3, 9):
            center = (1.+.03*(index-2), .5)
            p = front_packet(index, (index-1)*.1, center=center, depth_to_face=.15)
            boxes, _ = provider.update(p, (index-1)*.1)
            assert len(boxes) == 1
            assert boxes[0].xy == pytest.approx(center, abs=.008)
            assert provider.permission('B1', (index-1)*.1) is not None
            measured = provider.last_delivery['detections'][0]
            assert measured['geometry_source'] == 'current_lidar_bounded_single_face'
            assert measured['hidden_thickness_source'] == 'measured_registry'
            assert len(measured['visible_faces']) == 1
            assert all(count >= 3 for count in measured['boundary_free_ray_counts'])
            assert measured['scan_id'] == index
    finally:
        provider.close()


@pytest.mark.parametrize('options', [{'truncated': True}, {'sides': (-1,)}, {'sides': (1,)},
                                     {'sides': ()}, {'wall': True}])
def test_single_face_rejects_partial_edge_missing_free_side_or_wall_background(options):
    provider = engine()
    try:
        establish(provider)
        assert provider.update(front_packet(3, .2, **options), .2)[0] == ()
        assert provider.permission('B1', .2) is None
    finally:
        provider.close()


def test_bounded_face_never_creates_initial_identity_and_never_revives_old_hint():
    provider = engine()
    try:
        assert provider.update(front_packet(1, 0.), 0.)[0] == ()
        assert provider.update(front_packet(2, .1), .1)[0] == ()
        assert provider.permission('B1', .1) is None
        provider.reset()
        establish(provider)
        assert provider.update(front_packet(3, .4), .4)[0] == ()
        assert provider.permission('B1', .4) is None
    finally:
        provider.close()


def test_single_face_cannot_use_duplicate_or_out_of_height_rays_as_boundary_proof():
    provider = engine()
    try:
        establish(provider)
        points, origin = bounded_face_cloud()
        # Keep the full face, but duplicate only one ray on each boundary.
        face_count = 22*16
        duplicate_rays = np.repeat(points[[face_count, face_count+5]], 20, axis=0)
        p = packet(3, .2, np.vstack([points[:face_count], duplicate_rays]))
        p.observation['sensors']['lidar']['position'] = origin
        assert provider.update(p, .2)[0] == ()
    finally:
        provider.close()


def test_boundary_proof_requires_rays_stay_outside_and_pass_registered_far_face():
    from deploy.perception.box_fitting import _bounded_face_rays
    normal, tangent = np.array([-1., 0.]), np.array([0., 1.])
    edges, plane_offset = (-.15, .15), .2  # front face x=-.2
    def rays_from(origin, end_x):
        result = []
        for sign in (-1, 1):
            for z in (.25, .29, .33, .37, .41):
                crossing = np.array([-.2, sign*.19, z])
                result.append(origin+(end_x-origin[0])/(-.2-origin[0])*(crossing-origin))
        return np.asarray(result)
    origin = np.array([-.65, 0., .3])
    assert _bounded_face_rays(rays_from(origin, .45), origin, normal, tangent,
                             plane_offset, edges, .5, .4, DEFAULT_TRACKING) == (5, 5)
    # A background return only 10 cm behind the near face proves no clearance
    # around the registered 40 cm thickness.
    assert _bounded_face_rays(rays_from(origin, -.1), origin, normal, tangent,
                             plane_offset, edges, .5, .4, DEFAULT_TRACKING) is None
    # At an oblique view, a ray outside the front edge can cross the hidden box
    # volume further back. It is contradictory evidence, not edge clearance.
    origin = np.array([-.65, .7, .3])
    assert _bounded_face_rays(rays_from(origin, .45), origin, normal, tangent,
                             plane_offset, edges, .5, .4, DEFAULT_TRACKING) is None
