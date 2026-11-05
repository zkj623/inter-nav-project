"""Export one comparable CSV row per recorded semantic exploration run."""

import argparse
import csv
import json
import sys
from pathlib import Path


FIELDS = (
    'run_id', 'record_dir', 'started_at', 'git_commit', 'git_dirty',
    'scene', 'random_seed', 'target_query', 'detection_mode', 'qwen_active',
    'status', 'reason', 'step', 'last_recorded_step', 'last_recorded_source', 'target_confirmed',
    'arrival_verified', 'goal_distance_m', 'profile_goal_distance_min_m',
    'profile_goal_distance_final_m', 'open_vocabulary_frames',
    'open_vocabulary_failures', 'plans', 'replans',
    'voronoi_incremental_updates', 'voronoi_incremental_fallbacks',
    'artifacts_complete',
)


def _read_json(path):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _last_recorded_step(directory, summary, progress):
    if summary.get('status') not in (None, 'running') and isinstance(summary.get('step'), int):
        return summary['step'], 'summary'
    candidates = []
    for source, value in (
        ('summary', summary.get('step')),
        ('progress', progress.get('last_step')),
    ):
        if isinstance(value, int):
            candidates.append((value, source))
    for filename, source in (
        ('run.log', 'run.log'),
        ('groundingdino_detections.jsonl', 'groundingdino_detections.jsonl'),
    ):
        path = directory / filename
        if not path.is_file():
            continue
        with path.open(encoding='utf-8', errors='replace') as file:
            for line in file:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(item, dict):
                    continue
                if filename == 'run.log' and item.get('event') not in (
                    'semantic_exploration_progress', 'semantic_exploration_result',
                    'semantic_exploration_stopped',
                ):
                    continue
                value = item.get('step')
                if isinstance(value, int):
                    candidates.append((value, source))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])


def collect_runs(root: Path):
    directories = {
        path.parent for filename in ('run_summary.json', 'manifest.json')
        for path in root.rglob(filename)
    }
    for directory in sorted(directories):
        summary = _read_json(directory / 'run_summary.json')
        manifest = _read_json(directory / 'manifest.json')
        progress = _read_json(directory / 'progress.json')
        run_config = manifest.get('run_config') or {}
        profile = manifest.get('scene_profile') or {}
        perception = summary.get('perception') or {}
        planning = summary.get('planning') or {}
        voronoi = summary.get('voronoi') or {}
        git = manifest.get('git') or {}
        last_step, step_source = _last_recorded_step(directory, summary, progress)
        status = summary.get('status')
        # A startup-only "running" file is not evidence that the process is
        # still alive. Keep the raw status in run_summary.json for inspection.
        if status in (None, 'running'):
            status = 'unfinished_record'
        yield {
            'run_id': manifest.get('run_id') or summary.get('run_id') or directory.name,
            'record_dir': str(directory),
            'started_at': manifest.get('started_at') or summary.get('started_at'),
            'git_commit': git.get('commit'),
            'git_dirty': git.get('dirty'),
            'scene': profile.get('name'),
            'random_seed': manifest.get('random_seed'),
            'target_query': summary.get('target_query') or run_config.get('target_query'),
            'detection_mode': (perception.get('effective_mode') or
                               run_config.get('semantic_detection_mode') or
                               summary.get('detection_mode')),
            'qwen_active': manifest.get('qwen_active'),
            'status': status,
            'reason': summary.get('reason'),
            'step': summary.get('step'),
            'last_recorded_step': last_step,
            'last_recorded_source': step_source,
            'target_confirmed': summary.get('target_confirmed'),
            'arrival_verified': summary.get('arrival_verified'),
            'goal_distance_m': summary.get('goal_distance_m'),
            'profile_goal_distance_min_m': summary.get('profile_goal_distance_min_m'),
            'profile_goal_distance_final_m': summary.get('profile_goal_distance_final_m'),
            'open_vocabulary_frames': perception.get('open_vocabulary_frames'),
            'open_vocabulary_failures': perception.get('open_vocabulary_failures'),
            'plans': planning.get('plans'),
            'replans': planning.get('replans'),
            'voronoi_incremental_updates': voronoi.get('incremental_updates'),
            'voronoi_incremental_fallbacks': voronoi.get('incremental_fallbacks'),
            'artifacts_complete': summary.get('artifacts_complete'),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='directory containing run folders')
    parser.add_argument('--output', type=Path, help='write CSV here instead of stdout')
    args = parser.parse_args(argv)
    if args.output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(collect_runs(args.root))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(collect_runs(args.root))


if __name__ == '__main__':
    main()
