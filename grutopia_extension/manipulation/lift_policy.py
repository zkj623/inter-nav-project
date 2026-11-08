"""Supervised imitation of successful lifts; not an autonomous RL policy."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def make_network(observation_size, action_size):
    return nn.Sequential(
        nn.Linear(observation_size, 192), nn.Tanh(),
        nn.Linear(192, 192), nn.Tanh(),
        nn.Linear(192, 192), nn.Tanh(),
        nn.Linear(192, action_size),
    )


def train_policy(demonstrations, output, epochs=2500, seed=7):
    if not demonstrations or epochs <= 0:
        raise ValueError('successful demonstrations and positive epochs are required')
    if Path(output).exists():
        raise FileExistsError('Use a new checkpoint path; existing policies will not be overwritten.')
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    observations = []
    actions = []
    sources = []
    for path in demonstrations:
        path = Path(path)
        result_path = path.parent / 'result.json'
        if not result_path.exists() or not json.loads(result_path.read_text()).get('success'):
            raise ValueError(f'Refusing a failed or unverified demonstration: {path}')
        with np.load(path, allow_pickle=False) as data:
            episode_observations = data['observations'].copy()
            episode_actions = data['actions']
            if (episode_observations.ndim != 2 or episode_actions.ndim != 2
                    or episode_observations.shape[1] != 23 or episode_actions.shape[1] != 11
                    or len(episode_observations) == 0 or len(episode_observations) != len(episode_actions)):
                raise ValueError('This trainer requires single-arm 23-observation/11-action data; '
                                 'bimanual recordings need a matching training task.')
            episode_observations[:, 11:14] = episode_observations[0, 11:14]
            observations.append(episode_observations)
            actions.append(episode_actions)
        sources.append(str(path.resolve()))
    observations = np.concatenate(observations)
    actions = np.concatenate(actions)
    if len(observations) < 20 or not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError('demonstration contains too few samples or nonfinite data')
    order = np.random.default_rng(seed).permutation(len(observations))
    split = max(1, int(0.9 * len(order)))
    train_indices = order[:split]
    validation_indices = order[split:]
    observation_mean = torch.from_numpy(observations[train_indices].mean(axis=0))
    observation_scale = torch.from_numpy(np.maximum(observations[train_indices].std(axis=0), 0.05))
    action_mean = torch.from_numpy(actions[train_indices].mean(axis=0))
    action_scale = torch.from_numpy(np.maximum(actions[train_indices].std(axis=0), 0.03))
    inputs = (torch.from_numpy(observations) - observation_mean) / observation_scale
    feature_mask = torch.ones(observations.shape[1])
    feature_mask[:11] = 0.0
    feature_mask[13] = 0.0
    inputs = inputs * feature_mask
    targets = (torch.from_numpy(actions) - action_mean) / action_scale
    network = make_network(observations.shape[1], actions.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    best_error = float('inf')
    best_state = None
    completed = 0
    for epoch in range(epochs):
        network.train()
        prediction = network(inputs[train_indices])
        loss = torch.mean((prediction - targets[train_indices]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        completed = epoch + 1
        if epoch % 100 == 0 or epoch + 1 == epochs:
            network.eval()
            with torch.no_grad():
                validation_prediction = network(inputs[validation_indices])
                error = float(torch.mean((validation_prediction - targets[validation_indices]) ** 2))
            if error < best_error:
                best_error = error
                best_state = {name: tensor.detach().clone() for name, tensor in network.state_dict().items()}
            if epoch % 500 == 0 or epoch + 1 == epochs:
                print(json.dumps({'epoch': epoch + 1, 'training_loss': float(loss), 'validation_loss': error}), flush=True)
    network.load_state_dict(best_state)
    with torch.no_grad():
        predictions = network(inputs[validation_indices]) * action_scale + action_mean
        residual = predictions - torch.from_numpy(actions[validation_indices])
        rmse = torch.sqrt(torch.mean(residual ** 2, dim=0)).tolist()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'method': 'behavior_cloning', 'phase_scheduler': 'scripted',
        'observations': '11 joint positions, object origin xyz, dimensions xyz, phase one-hot, phase progress',
        'privileged_object_pose': True, 'samples': len(observations), 'demonstrations': sources,
        'policy_inputs_used': 'initial object xy, body dimensions, prescribed phase, phase progress',
        'policy_scope': 'open-loop learned joint trajectory with feedback joint drives; not a state-feedback grasp policy',
        'epochs': completed, 'seed': seed, 'validation_normalized_mse': best_error,
        'validation_action_rmse': rmse,
        'validation_scope': 'held-out time samples, not held-out objects or episodes; physical rollout required',
    }
    torch.save({'state_dict': best_state, 'observation_mean': observation_mean,
                'observation_scale': observation_scale, 'action_mean': action_mean,
                'action_scale': action_scale, 'feature_mask': feature_mask, 'metadata': metadata}, output)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def load_policy(path):
    torch.set_num_threads(2)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    network = make_network(len(checkpoint['observation_mean']), len(checkpoint['action_mean']))
    network.load_state_dict(checkpoint['state_dict'])
    network.eval()

    def predict(observation):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        normalized = (observation - checkpoint['observation_mean']) / checkpoint['observation_scale']
        normalized = normalized * checkpoint.get('feature_mask', torch.ones_like(normalized))
        with torch.no_grad():
            action = network(normalized) * checkpoint['action_scale'] + checkpoint['action_mean']
        return action.numpy()

    return predict
