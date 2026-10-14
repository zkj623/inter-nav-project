"""Compact, explicit outcome record for a semantic exploration run."""

import json
import math
from pathlib import Path


def summary_path(record_dir: str, map_output: str):
    if record_dir:
        return Path(record_dir) / 'run_summary.json'
    if map_output:
        return Path(str(map_output) + '_run_summary.json')
    return None


def write_run_summary(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def build_run_summary(
    *,
    target_query: str,
    detection_mode: str,
    max_steps: int,
    profile_goal=None,
    component=None,
    terminal_result=None,
    step=None,
    robot_observation=None,
    stop_reason: str = 'incomplete',
    error_type=None,
    exit_code=None,
):
    if terminal_result is not None:
        position = terminal_result.position
    elif robot_observation is not None:
        position = robot_observation.get('position')
    else:
        position = None
    if position is None and component is not None and component._trajectory:
        position = component._trajectory[-1]
    position = None if position is None else [float(value) for value in position[:3]]

    # The scene profile's goal is a benchmark reference, never an input to the
    # online semantic decision. Preserve closest and final proximity separately.
    profile_goal = None if profile_goal is None else [float(value) for value in profile_goal[:3]]
    profile_distances = (
        [] if profile_goal is None or component is None else [
            math.dist(point[:2], profile_goal[:2]) for point in component._trajectory
        ]
    )
    profile_closest_step = (
        None if not profile_distances else min(range(len(profile_distances)), key=profile_distances.__getitem__)
    )
    profile_final_distance = (
        None if profile_goal is None or position is None
        else math.dist(position[:2], profile_goal[:2])
    )
    profile_min_distance = (
        min(profile_distances) if profile_distances else profile_final_distance
    )

    goal = None if component is None else component.target_navigation_position
    goal = None if goal is None else [float(value) for value in goal]
    node = None if component is None else component.target_node
    confirmed = False if component is None else component.target_confirmed
    threshold = None if component is None else float(component.config.target_approach_tolerance)
    target_distance = (
        None if position is None or node is None
        else math.dist(position[:2], node.position[:2])
    )
    distance = (
        None if position is None or goal is None
        else math.dist(position[:2], goal[:2])
    )
    arrived = (
        terminal_result is not None
        and terminal_result.success
        and confirmed
        and distance is not None
        and threshold is not None
        and distance <= threshold
    )
    if arrived:
        status, reason = 'succeeded', 'goal_reached'
    elif terminal_result is not None and terminal_result.terminal:
        status = 'failed'
        reason = terminal_result.failure_reason or (
            'target_not_confirmed' if not confirmed else 'arrival_not_verified'
        )
    elif error_type is not None:
        status, reason = 'failed', stop_reason
    else:
        status, reason = 'incomplete', stop_reason

    stats = (
        terminal_result.statistics
        if terminal_result is not None and terminal_result.statistics
        else ({} if component is None else component.statistics())
    )
    voronoi = stats.get('semantic_voronoi', {})
    return {
        'schema_version': 2,
        'status': status,
        'reason': reason,
        'arrival_verified': bool(arrived),
        'step': step,
        'max_steps': max_steps,
        'exit_code': exit_code,
        'error_type': error_type,
        'target_query': target_query,
        'target_found': node is not None,
        'target_confirmed': bool(confirmed),
        'target_label_support': (
            0 if component is None else component._target_label_support()[0]
        ),
        'target_match_method': (
            None if node is None else ('lexical' if component._target_lexical else 'embedding')
        ),
        'rejected_provisional_targets': (
            0 if component is None else len(component._rejected_embedding_targets)
        ),
        'target_node': None if node is None else node.node_id,
        'target_label': None if node is None else node.label,
        'target_position': None if node is None else [float(value) for value in node.position],
        'approach_goal': goal,
        'final_position': position,
        'goal_distance_m': None if distance is None else round(distance, 4),
        'target_distance_m': None if target_distance is None else round(target_distance, 4),
        'arrival_threshold_m': threshold,
        'arrival_margin_m': (
            None if distance is None or threshold is None else round(threshold - distance, 4)
        ),
        'arrival_rule': 'xy_distance_to_approach_goal',
        'profile_goal': profile_goal,
        'profile_goal_distance_min_m': (
            None if profile_min_distance is None else round(profile_min_distance, 4)
        ),
        'profile_goal_distance_min_step': profile_closest_step,
        'profile_goal_distance_final_m': (
            None if profile_final_distance is None else round(profile_final_distance, 4)
        ),
        'perception': {
            'requested_mode': detection_mode,
            'effective_mode': stats.get('semantic_detection_effective_mode'),
            'open_vocabulary_frames': stats.get('open_vocabulary_frames'),
            'open_vocabulary_failures': stats.get('open_vocabulary_failures'),
        },
        'planning': {
            'plans': stats.get('plans'),
            'replans': stats.get('replans'),
            'planning_failures': stats.get('planning_failures'),
        },
        'voronoi': {
            key: voronoi.get(key) for key in (
                'incremental_updates',
                'incremental_fallbacks',
                'incremental_fallback_reasons',
                'incremental_seam_checks',
                'incremental_seam_mismatches',
                'incremental_verification_checks',
                'incremental_verification_mismatches',
                'incremental_verification_mismatched_cells',
                'last_verification_mismatch_revision',
                'incremental_window_evaluations',
                'incremental_max_window_fraction',
            )
        } | {
            'window_to_changed_ratio': (
                None if not voronoi.get('incremental_changed_cells_total')
                else round(
                    voronoi.get('incremental_window_cells_total', 0)
                    / voronoi['incremental_changed_cells_total'],
                    2,
                )
            ),
            'incremental_seconds': round(voronoi.get('incremental_seconds', 0.0), 3),
            'full_build_seconds': round(voronoi.get('full_build_seconds', 0.0), 3),
            'verification_seconds': round(voronoi.get('verification_seconds', 0.0), 3),
        },
    }
