"""Local Codex/Astra generation and remote, frozen-policy multi-layout evaluation."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from grutopia_extension.interaction_clearance.benchmark import default_suite, digest, pose_suite, select_best, validate_suite
from grutopia_extension.interaction_clearance.skill_runtime import validate_skill_source


OUTPUT_SCHEMA = {'type': 'object', 'properties': {'source': {'type': 'string'}, 'rationale': {'type': 'string'}},
                 'required': ['source', 'rationale'], 'additionalProperties': False}


def training_feedback(candidates):
    return [{'candidate_id': item['candidate_id'], 'source': item['source'],
             'metrics': item['train']['metrics'],
             'cases': [{'layout_id': case['layout_id'], 'success': case['success'], 'reason': case['reason'],
                        'safety_failure': case['safety_failure'], 'policy_details': case.get('policy_details', {}),
                        'simulated_seconds': case['simulated_seconds']} for case in item['train']['results']]}
            for item in candidates]


def generation_prompt(suite, candidates, round_index, rejections=None):
    training_layouts = [item for item in suite['layouts'] if item['split'] == 'train']
    context = {'round': round_index, 'training_layouts': training_layouts, 'feedback': training_feedback(candidates),
               'source_rejections': rejections or []}
    if suite.get('policy_interface') == 'composable_v1':
        return primitive_generation_prompt(context)
    return '''Generate one trusted Python orchestration skill for mandatory robot blockage clearance.
Do not call tools, inspect files, edit the repository, or launch simulation. Return only JSON matching the supplied schema.
The action policy is FROZEN: pick_place(task) performs approach, squeeze, lift, local carry,
lower, release, retract, return, and 2 seconds settling. It cannot push, replan around obstacles,
choose an alternate placement target, or resume a failed grasp. Do not invent these capabilities.
Your skill must be a single undecorated def run(ctx), returning the Outcome from a ctx call.
Public API:
  ctx.execute('pick_place', object_id=ctx.task.object_id, place_xy=ctx.task.place_xy)
  ctx.verify_clearance() -> Outcome(success, reason, details)
  ctx.observe() -> a fresh dict with joint_positions, base_pose_world, object_pose_world,
    object_pose_in_base, placement_goal_in_base, measured_body_velocity,
    object_linear_velocity, object_angular_velocity, hand_normal_force_vectors
  ctx.wait(seconds) -> Outcome; physically holds current joints and stops wheel targets.
    Max 2 seconds per call, 6 seconds total. Every step retains all collision checks.
  ctx.report(message) -> event only.
Outcome fields are .success, .reason, .details. Frozen task geometry cannot be changed.
Policy call budget is 3. A failed policy locks further actions unless details['retry_safe'] is explicitly true.
Current frozen policy does NOT advertise retry_safe. Return failures; do not blindly retry.
Independent verification is authoritative. Geometry failures require later policy work, not fabricated skill success.
Allowed: local-name assignments, if/return, arithmetic, reading dict keys and Outcome fields,
literal for loops range(0), range(1), range(2), range(3), at most two nesting levels.
Only allowed function calls: the ctx methods above, range/len/min/max/abs/float/int/bool/tuple.
No comments, imports, helpers, annotations, decorators, while, comprehensions, private attributes,
assignment to attributes/subscripts, exponentiation, or more than 300 AST nodes / 16 KiB source.
Do not branch on layout IDs, seeds, candidate IDs or memorize scenario coordinates.
Use generic measured conditions. Prefer simpler, faster code if clearance is already reliable.
Optimize safety first, then worst-layout success, macro-average success, then successful execution time.
Only training feedback is provided; validation and held-out geometry are intentionally unavailable.
If orchestration cannot improve a failure, explain that limitation in rationale rather than pretending it can.
CONTEXT_JSON:
''' + json.dumps(context, ensure_ascii=False)


def primitive_generation_prompt(context):
    return '''Generate one generic Python skill that clears the upstream-designated blockage.
Return only JSON with string source and rationale. Do NOT use tools, inspect files or change the repository.
One undecorated def run(ctx) only; return an actual Outcome from a ctx call.
The immutable physical policy registry is now COMPOSABLE:
  ctx.execute('base_move', target='grasp'|'placement'|'home')
    goal grasp: derives robot XY/yaw from measured box pose and task grasp distance.
    goal placement: moves the physically held box to the exact upstream placement target, no rotation while carrying.
    goal home: returns to actual starting base pose AFTER release/retract.
    Paths are collision-checked using full robot/object envelopes before any motion.
    The base policy automatically searches a bounded SE(2) grid with A* when empty,
    or an XY grid at fixed yaw when carrying, using full robot and held-object collision envelopes.
    It certifies and shortcuts the path, then follows it with wheel feedback. Do not choose legacy route names.
    Planning rejection is retry_safe;
    collision, grasp loss, timeouts and execution failures stop further actuation.
  ctx.execute('arm_plan', mode='grasp'|'place')
    solves left/right tool poses in measured box frame, joint limits and collision penalties,
    then independently checks interpolated robot, object and walls. It executes NO motion.
    grasp requires base alignment; place requires held box within 5.5cm of exact target.
  ctx.execute('grasp_lift') consumes a fresh grasp plan; physically approaches, squeezes, lifts, checks bilateral force.
  ctx.execute('place_release') consumes fresh place plan; lowers, releases, retracts, settles and checks ground/contact.
State transitions ready -> holding -> released. Wrong stage returns a precondition error.
Every base action invalidates the cached arm plan. The cached plan also checks measured robot/box/joint drift.
Outcome fields: .success, .reason, .details. For BASE_PLAN_REJECTED and ARM_PLAN_REJECTED,
no motion happened and .details['retry_safe'] is true. Never retry execution failures blindly.
ctx.observe() -> fresh dict with base_pose_world, object_pose_world, object_pose_in_base,
placement_goal_in_base, measured_body_velocity, object_linear_velocity, object_angular_velocity,
joint_positions, hand_normal_force_vectors, interaction_state, plan_mode, safety_ok.
ctx.verify_clearance() -> authoritative Outcome; does not advance simulation.
ctx.wait(seconds) -> collision-checked hold, max 2 seconds/call and 6 seconds total.
ctx.report(message) logs only. The action call budget is 12, control-time budget 150 seconds.
Keep the safe sequence: base grasp alignment -> arm grasp planning -> grasp_lift -> base placement
-> arm place planning -> place_release -> base home -> independent verification.
Do not blindly repeat a deterministic no-path result: changing route names is no longer a skill responsibility.
Home return is part of the interaction completion protocol. Do not omit it merely to reduce reported time.
No direct joint/pose edits, attachment, teleport, scorer modification, pushing, learned policy updates or arbitrary end-effector targets.
No layout ID, seed, coordinates or individual-scenario memorization. Policy targets are derived from measured/task geometry.
Allowed local-name assignment, if/return, tuple/list literals, arithmetic, reading public fields/dict keys,
literal for range(0..3) loops with at most two levels; <=300 AST nodes, <=16KiB source.
Only callable functions: the listed ctx methods, range/len/min/max/abs/float/int/bool/tuple.
No comments, imports, helpers, nested functions, annotations, private attributes, comprehensions,
while, exponentiation or mutation of attributes/subscripts. Return policy failures unchanged.
Optimize safety, worst-layout and macro success, then time and action calls. Avoid unsupported recovery promises.
Only TRAINING layouts and feedback follow. Validation and held-out are intentionally excluded.
If no better behavior is supported, keep the incumbent exactly and explain why in rationale.
CONTEXT_JSON:
''' + json.dumps(context, ensure_ascii=False)


def generate_astra(codex, model, schema, prompt, folder, timeout):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'prompt.txt').write_text(prompt, encoding='utf-8')
    command = [codex, 'exec', '--model', model, '--ephemeral', '--skip-git-repo-check', '--sandbox', 'read-only',
               '-c', 'approval_policy="never"', '-c', 'model_reasoning_effort="high"', '--color', 'never',
               '--json', '--output-schema', str(schema), '-']
    completed = subprocess.run(command, input=prompt, capture_output=True, text=True, encoding='utf-8',
                               errors='replace', timeout=timeout)
    messages, usage, errors, tools_used = [], {}, [], False
    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'item.completed' and event.get('item', {}).get('type') == 'agent_message':
            messages.append(event['item'].get('text', ''))
        elif event.get('type') == 'turn.completed':
            usage = event.get('usage', {})
        elif event.get('type') in ('error', 'turn.failed'):
            error = event.get('error', {})
            errors.append(event.get('message', error.get('message', 'generation failed') if isinstance(error, dict) else str(error)))
        if event.get('type') in ('item.started', 'item.completed') and event.get('item', {}).get('type') in (
                'command_execution', 'file_change', 'mcp_tool_call', 'web_search'):
            tools_used = True
    metadata = {'model': model, 'returncode': completed.returncode, 'usage': usage,
                'authentication': 'inherited_local_codex_configuration', 'credential_exported': False,
                'tools_used': tools_used}
    (folder / 'generation_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if completed.returncode or not messages:
        raise RuntimeError('Astra call failed: ' + '; '.join(errors)[:1000])
    payload = json.loads(messages[-1])
    (folder / 'generation.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    if tools_used:
        raise ValueError('Generator used tools despite the pure code-generation contract')
    if not isinstance(payload, dict) or not isinstance(payload.get('source'), str) or not isinstance(payload.get('rationale'), str):
        raise ValueError('Generator must return string source and rationale fields')
    validate_skill_source(payload['source'])
    (folder / 'candidate.py').write_text(payload['source'], encoding='utf-8', newline='\n')
    return payload


class RemoteEvaluator:
    def __init__(self, host, repo, remote_output, local_output, gpus, timeout):
        self.host, self.repo, self.remote_output = host, repo, remote_output
        self.local_output, self.gpus, self.timeout = local_output, gpus, timeout
        self.prefix = ('set -eo pipefail\ncd ' + shlex.quote(repo) +
                       '\nsource /data20t/embodied/share/inter-nav/env.sh\n'
                       'export PYTHONPATH=/data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps:$PWD:$PYTHONPATH\n')
        self.frozen = None

    def ssh(self, script, timeout=60):
        encoded = base64.b64encode(script.encode()).decode()
        command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.host,
                   f'printf %s {encoded} | base64 -d | bash']
        completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
        if completed.returncode:
            raise RuntimeError('Remote command failed: ' + completed.stderr[-1500:])
        return completed.stdout

    def upload(self, source, destination):
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                        str(source), f'{self.host}:{destination}'], check=True, timeout=60)

    def prepare(self, suite_path):
        self.ssh('set -e\nif [ -e ' + shlex.quote(self.remote_output) + ' ]; then echo "Use a new remote output path" >&2; exit 1; fi\n'
                 'mkdir -p ' + shlex.quote(self.remote_output))
        self.upload(suite_path, self.remote_output + '/suite.json')
        output = self.ssh(self.prefix + '"$ISAAC_PYTHON" grutopia/demo/r1lite_clearance_multilayout.py --fingerprint-only\n')
        self.frozen = json.loads(output.splitlines()[-1])
        (self.local_output / 'runtime_manifest.json').write_text(json.dumps(self.frozen, indent=2))

    def evaluate(self, candidate_path, candidate_id, split):
        remote_skill = self.remote_output + '/' + candidate_id + '.py'
        self.upload(candidate_path, remote_skill)
        remote_run = self.remote_output + '/' + candidate_id + '_' + split
        gpu_flags = ' '.join(str(value) for value in self.gpus)
        command = ('"$ISAAC_PYTHON" grutopia/demo/r1lite_clearance_multilayout.py --suite ' +
                   shlex.quote(self.remote_output + '/suite.json') + ' --split ' + split + ' --skill ' + shlex.quote(remote_skill) +
                   ' --output ' + shlex.quote(remote_run) + ' --gpus ' + gpu_flags + ' --episode-timeout ' + str(self.timeout) +
                   ' --expected-runtime ' + self.frozen['sha256'])
        script = (self.prefix + 'if ! ' + command + ' > ' + shlex.quote(remote_run + '.log') + ' 2>&1; then tail -60 ' +
                  shlex.quote(remote_run + '.log') + ' >&2; exit 1; fi\ncat ' + shlex.quote(remote_run + '/evaluation.json'))
        print(json.dumps({'event': 'evaluation_begin', 'candidate_id': candidate_id, 'split': split, 'gpus': self.gpus}), flush=True)
        output = self.ssh(script, timeout=7200)
        result = json.loads(output)
        result['candidate_id'] = candidate_id
        result['remote_run'] = remote_run
        (self.local_output / f'{candidate_id}_{split}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'event': 'evaluation_complete', 'candidate_id': candidate_id, 'split': split,
                          'metrics': result['metrics']}), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-output', required=True)
    parser.add_argument('--host', default='zkj@162.105.195.166')
    parser.add_argument('--repo', default='/data20t/embodied/zkj/inter-nav-project')
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--pose-primitives', action='store_true')
    parser.add_argument('--baseline-skill', type=Path)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--gpus', nargs='+', type=int, default=[3, 7])
    parser.add_argument('--episode-timeout', type=int, default=900)
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--codex', default=shutil.which('codex.cmd') or shutil.which('codex'))
    parser.add_argument('--generation-timeout', type=int, default=240)
    args = parser.parse_args()
    if not args.codex:
        parser.error('Local Codex CLI is required; it inherits existing provider/auth settings')
    if not 1 <= args.rounds <= 10:
        parser.error('Use 1..10 generation rounds')
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Use a new empty local output directory')
    suite = validate_suite(json.loads(args.suite.read_text()) if args.suite else (pose_suite() if args.pose_primitives else default_suite()))
    args.output.mkdir(parents=True, exist_ok=True)
    suite_path = args.output / 'suite.json'
    suite_path.write_text(json.dumps(suite, indent=2))
    schema = args.output / 'generation_schema.json'
    schema.write_text(json.dumps(OUTPUT_SCHEMA, indent=2))
    evaluator = RemoteEvaluator(args.host, args.repo, args.remote_output, args.output, args.gpus, args.episode_timeout)
    evaluator.prepare(suite_path)
    root = Path(__file__).resolve().parents[3]
    default_skill = 'composable_clear_blockage.py' if suite.get('policy_interface') == 'composable_v1' else 'clear_blockage.py'
    baseline = (args.baseline_skill or root / 'grutopia_extension/interaction_clearance/skills' / default_skill).read_text(encoding='utf-8')
    validate_skill_source(baseline)
    baseline_path = args.output / 'baseline.py'
    baseline_path.write_text(baseline, encoding='utf-8', newline='\n')
    candidates = [{'candidate_id': 'baseline', 'source': baseline, 'path': str(baseline_path),
                   'train': evaluator.evaluate(baseline_path, 'baseline', 'train')}]
    rejections = []
    history_path = args.output / 'training_history.json'
    history_path.write_text(json.dumps({'candidates': candidates, 'rejections': rejections}, indent=2))
    for round_index in range(1, args.rounds + 1):
        candidate_id = f'astra_{round_index:02d}'
        prompt = generation_prompt(suite, candidates, round_index, rejections)
        folder = args.output / candidate_id
        print(json.dumps({'event': 'generation_begin', 'round': round_index, 'model': args.model}), flush=True)
        try:
            generated = generate_astra(args.codex, args.model, schema, prompt, folder, args.generation_timeout)
        except (ValueError, SyntaxError, TypeError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
            rejection = {'round': round_index, 'reason': type(error).__name__, 'error': str(error)}
            rejections.append(rejection)
            (folder / 'rejection.json').write_text(json.dumps(rejection, indent=2))
            history_path.write_text(json.dumps({'candidates': candidates, 'rejections': rejections}, indent=2))
            print(json.dumps({'event': 'candidate_rejected', **rejection}), flush=True)
            continue
        candidate_path = folder / 'candidate.py'
        identity = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        duplicate = next((item for item in candidates if item['train']['skill_sha256'] == identity), None)
        if duplicate:
            train = dict(duplicate['train'], candidate_id=candidate_id, reused_identical_evaluation=True)
            print(json.dumps({'event': 'duplicate_candidate', 'candidate_id': candidate_id}), flush=True)
        else:
            train = evaluator.evaluate(candidate_path, candidate_id, 'train')
        candidates.append({'candidate_id': candidate_id, 'source': generated['source'], 'rationale': generated['rationale'],
                           'path': str(candidate_path), 'train': train})
        history_path.write_text(json.dumps({'candidates': candidates, 'rejections': rejections}, indent=2))
    ranked = sorted(candidates, key=lambda item: tuple(item['train']['metrics']['selection_score']), reverse=True)
    shortlist = []
    seen = set()
    for candidate in ranked:
        identity = candidate['train']['skill_sha256']
        if identity not in seen:
            shortlist.append(candidate)
            seen.add(identity)
        if len(shortlist) == 2:
            break
    validation = [evaluator.evaluate(Path(item['path']), item['candidate_id'], 'validation') for item in shortlist]
    selected = select_best(validation)
    champion = next(item for item in shortlist if item['candidate_id'] == selected['candidate_id'])
    (args.output / 'selected_skill.py').write_bytes(Path(champion['path']).read_bytes())
    selection = {'candidate_id': champion['candidate_id'], 'skill_sha256': selected['skill_sha256'],
                 'validation': selected, 'heldout_not_used_for_selection': True}
    (args.output / 'selection_before_heldout.json').write_text(json.dumps(selection, indent=2))
    heldout = evaluator.evaluate(Path(champion['path']), champion['candidate_id'], 'heldout')
    report = {'suite_sha256': digest(suite), 'runtime_sha256': evaluator.frozen['sha256'], 'rounds_requested': args.rounds,
              'astra_candidates_valid': len(candidates) - 1, 'rejections': rejections,
              'selected_candidate': champion['candidate_id'], 'selected_skill_sha256': selected['skill_sha256'],
              'baseline_train_metrics': candidates[0]['train']['metrics'], 'selected_train_metrics': champion['train']['metrics'],
              'validation_metrics': selected['metrics'], 'heldout_metrics': heldout['metrics'],
              'improved_training_score': tuple(champion['train']['metrics']['selection_score']) > tuple(candidates[0]['train']['metrics']['selection_score']),
              'policy_frozen': True, 'policy_training_performed': False, 'backend': 'isaac_physx',
              'model': args.model, 'credential_exported': False, 'heldout_feedback_sent_to_generator': False,
              'statistical_reliability_established': False, 'remote_output': args.remote_output}
    report['policy_interface'] = suite.get('policy_interface', 'legacy_pick_place')
    report['initial_pose_variations'] = suite.get('policy_interface') == 'composable_v1'
    (args.output / 'evolution_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'event': 'evolution_complete', **report}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
