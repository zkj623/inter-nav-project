"""Versioned skill events, state-action transitions, and clearance videos."""

import json
from pathlib import Path

import numpy as np


PHASES = ('approach', 'squeeze', 'lift', 'hold', 'transport', 'lower', 'release', 'retract', 'return', 'settle', 'base_move')


class EpisodeRecorder:
    def __init__(self, output, task, skill_hash):
        import cv2
        from grutopia_extension.interaction_clearance.primitives import VERSION

        self.cv2 = cv2
        self.output = Path(output)
        self.events = (self.output / 'events.jsonl').open('w', encoding='utf-8')
        self.transitions = (self.output / 'transitions.jsonl').open('w', encoding='utf-8')
        self.overview = cv2.VideoWriter(str(self.output / 'clearance_overview.mp4'),
                                        cv2.VideoWriter_fourcc(*'mp4v'), 50, (1280, 720))
        self.layout = cv2.VideoWriter(str(self.output / 'clearance_layout.mp4'),
                                      cv2.VideoWriter_fourcc(*'mp4v'), 50, (960, 720))
        if not self.overview.isOpened() or not self.layout.isOpened():
            raise RuntimeError('Could not create clearance video writers')
        self.observations = []
        self.actions = []
        self.phase_indices = []
        self.metadata = {'format_version': 2, 'task': task.to_dict(), 'skill_sha256': skill_hash,
                         'policy_id': 'registry', 'policy_version': VERSION,
                         'training_performed': False, 'observation_source': 'simulation_ground_truth',
                         'observation_size': 42 + len(PHASES), 'action_size': 22, 'phases': list(PHASES),
                         'observation_fields': ['joint_positions:19', 'object_pose_in_base:7',
                                                'placement_goal_in_base:3', 'measured_body_velocity:3',
                                                'finger_normal_forces:6', 'object_dimensions:3',
                                                f'phase_onehot:{len(PHASES)}', 'phase_progress:1'],
                         'action_fields': ['joint_position_targets:19', 'base_body_twist_targets:3'],
                         'control_hz': 50, 'physics_hz': 200}

    def event(self, kind, details):
        self.events.write(json.dumps({'event': kind, **details}, ensure_ascii=False) + '\n')
        self.events.flush()

    def capture(self, environment, phase, progress, before, action):
        after = environment.observe()
        encoding = np.zeros(len(PHASES))
        encoding[PHASES.index(phase)] = 1
        observation = np.r_[before['joint_positions'], before['object_pose_in_base'],
                            before['placement_goal_in_base'], before['measured_body_velocity'],
                            np.asarray(before['hand_normal_force_vectors']).ravel(),
                            environment.task.dimensions, encoding, progress]
        if observation.shape != (self.metadata['observation_size'],) or np.asarray(action).shape != (22,):
            raise RuntimeError('Unexpected clearance state-action schema')
        self.observations.append(observation)
        self.actions.append(action)
        self.phase_indices.append(PHASES.index(phase))
        self.transitions.write(json.dumps({'step': len(self.actions) - 1, 'phase': phase,
                                           'progress': progress, 'observation': before,
                                           'action': list(map(float, action)), 'next_observation': after}) + '\n')
        evaluation = environment.evaluate()
        gap = environment.physics_audit[-1]['minimum_noncontact_clearance_m']
        label = f'CLEARANCE | {phase} | road {"CLEAR" if evaluation["road_clear"] else "BLOCKED"} | gap {gap * 1000:.1f} mm'
        for close, writer in ((False, self.overview), (True, self.layout)):
            frame = environment.image(close=close)
            self.cv2.putText(frame, label, (18, 35), self.cv2.FONT_HERSHEY_SIMPLEX, 0.56, (250, 250, 250), 2)
            writer.write(frame)

    def snapshot(self, environment, name):
        self.cv2.imwrite(str(self.output / f'{name}.jpg'), environment.image())
        self.cv2.imwrite(str(self.output / f'{name}_layout.jpg'), environment.image(close=True))

    def close(self, result):
        self.overview.release()
        self.layout.release()
        self.events.close()
        self.transitions.close()
        eligible = bool(result.get('success') and result.get('collision_audit_passed'))
        self.metadata.update({'training_eligible': eligible, 'failure_reason': result.get('reason'),
                              'frames': len(self.actions)})
        np.savez_compressed(self.output / 'demonstration.npz',
                            observations=np.asarray(self.observations, dtype=np.float32).reshape(-1, self.metadata['observation_size']),
                            actions=np.asarray(self.actions, dtype=np.float32).reshape(-1, 22),
                            phase_indices=np.asarray(self.phase_indices, dtype=np.int32),
                            training_eligible=np.array(eligible))
        (self.output / 'dataset_metadata.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')
