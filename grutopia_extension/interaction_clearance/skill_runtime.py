"""Bounded skill programs call public policy APIs, never simulator objects."""

import ast
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path


@dataclass(frozen=True)
class Outcome:
    success: bool
    reason: str
    details: dict = field(default_factory=dict)


class SkillContext:
    def __init__(self, task, policies, verify, observe, event, maximum_calls=12, wait=None):
        self.task = task
        self._policies = policies
        self._verify = verify
        self._observe = observe
        self._event = event
        self._maximum_calls = maximum_calls
        self._calls = 0
        self._wait = wait
        self._waited_seconds = 0.0
        self._stopped = False
        self.executed_policy_ids = []

    def execute(self, policy_id, *, object_id=None, place_xy=None, target=None, route=None, mode=None):
        if self._stopped:
            return Outcome(False, 'UNSAFE_RETRY_REJECTED')
        if self._calls >= self._maximum_calls:
            return Outcome(False, 'POLICY_CALL_BUDGET_EXCEEDED')
        if policy_id not in self._policies:
            return Outcome(False, 'UNKNOWN_POLICY')
        options = {}
        if policy_id == 'pick_place':
            if object_id != self.task.object_id or place_xy is None or tuple(place_xy) != tuple(self.task.place_xy):
                return Outcome(False, 'UPSTREAM_TARGET_MISMATCH')
            if any(value is not None for value in (target, route, mode)):
                return Outcome(False, 'INVALID_POLICY_ARGUMENTS')
        elif policy_id == 'base_move':
            if target not in ('grasp', 'placement', 'home') or (route is not None and route not in ('auto', 'direct', 'retreat', 'x_first', 'y_first')):
                return Outcome(False, 'INVALID_POLICY_ARGUMENTS')
            if object_id is not None or place_xy is not None or mode is not None:
                return Outcome(False, 'INVALID_POLICY_ARGUMENTS')
            options = {'target': target, 'route': route or 'auto'}
        elif policy_id == 'arm_plan':
            if mode not in ('grasp', 'place') or any(value is not None for value in (object_id, place_xy, target, route)):
                return Outcome(False, 'INVALID_POLICY_ARGUMENTS')
            options = {'mode': mode}
        elif any(value is not None for value in (object_id, place_xy, target, route, mode)):
            return Outcome(False, 'INVALID_POLICY_ARGUMENTS')
        self._calls += 1
        self.executed_policy_ids.append(policy_id)
        self._event('policy_begin', {'policy_id': policy_id, 'object_id': object_id,
                                     'place_xy': list(place_xy) if place_xy is not None else None,
                                     'options': options, 'attempt': self._calls})
        try:
            outcome = self._policies[policy_id].execute(self.task, **options)
            if not isinstance(outcome, Outcome):
                outcome = Outcome(False, 'INVALID_POLICY_OUTCOME')
        except Exception as error:
            outcome = Outcome(False, 'POLICY_EXCEPTION', {'error': str(error)})
        if not outcome.success and not outcome.details.get('retry_safe', False):
            self._stopped = True
        self._event('policy_end', {'policy_id': policy_id, 'success': outcome.success,
                                   'reason': outcome.reason, 'details': outcome.details})
        return outcome

    def verify_clearance(self):
        return self._verify()

    def observe(self):
        return self._observe()

    def report(self, message):
        self._event('skill_message', {'message': str(message)[:1024]})

    def wait(self, seconds):
        if self._stopped:
            return Outcome(False, 'SAFETY_STOPPED')
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= 2.0:
            return Outcome(False, 'INVALID_WAIT_DURATION')
        if self._waited_seconds + seconds > 6.0:
            return Outcome(False, 'WAIT_BUDGET_EXCEEDED')
        if self._wait is None:
            return Outcome(False, 'WAIT_UNAVAILABLE')
        self._waited_seconds += seconds
        try:
            outcome = self._wait(seconds)
            if not isinstance(outcome, Outcome):
                outcome = Outcome(False, 'INVALID_WAIT_OUTCOME')
        except Exception as error:
            outcome = Outcome(False, 'WAIT_EXCEPTION', {'error': str(error)})
        if not outcome.success:
            self._stopped = True
        self._event('skill_wait', {'seconds': seconds, 'success': outcome.success, 'reason': outcome.reason})
        return outcome


def validate_skill_source(source):
    if len(source.encode('utf-8')) > 16384:
        raise ValueError('Skill source exceeds 16 KiB')
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    if len(nodes) > 300:
        raise ValueError('Skill exceeds the AST node budget')
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1 or functions[0].name != 'run':
        raise ValueError('Skill must contain exactly def run(ctx)')
    arguments = functions[0].args
    if (functions[0].decorator_list or len(arguments.args) != 1 or arguments.args[0].arg != 'ctx'
            or arguments.posonlyargs or arguments.kwonlyargs or arguments.defaults
            or arguments.vararg or arguments.kwarg or functions[0].returns):
        raise ValueError('Skill entry point must be undecorated run(ctx)')
    forbidden = (ast.Import, ast.ImportFrom, ast.While, ast.ClassDef, ast.Lambda, ast.AsyncFunctionDef,
                 ast.With, ast.AsyncWith, ast.Global, ast.Nonlocal, ast.ListComp, ast.DictComp,
                 ast.SetComp, ast.GeneratorExp, ast.Delete, ast.Await, ast.Yield, ast.YieldFrom)
    allowed_calls = {'range', 'len', 'min', 'max', 'abs', 'float', 'int', 'bool', 'tuple'}
    allowed_methods = {'execute', 'verify_clearance', 'observe', 'report', 'wait'}
    for node in nodes:
        if isinstance(node, forbidden):
            raise ValueError(f'Forbidden skill syntax: {type(node).__name__}')
        if isinstance(node, ast.FunctionDef) and node is not functions[0]:
            raise ValueError('Nested functions are not supported')
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('Private attributes are inaccessible to skill code')
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(not isinstance(target, ast.Name) for target in targets):
                raise ValueError('Skill assignments must target local names')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_calls:
                    raise ValueError(f'Forbidden skill call: {node.func.id}')
            elif not (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                      and node.func.value.id == 'ctx' and node.func.attr in allowed_methods):
                raise ValueError('Only whitelisted ctx methods may be called')
        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name):
                raise ValueError('Skill loop targets must be local names')
            if any(isinstance(descendant, ast.For) and any(isinstance(nested, ast.For)
                   for nested in ast.walk(descendant) if nested is not descendant)
                   for descendant in ast.walk(node) if descendant is not node):
                raise ValueError('Skill loop nesting exceeds two levels')
            iterator = node.iter
            if not (isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name)
                    and iterator.func.id == 'range' and len(iterator.args) == 1
                    and isinstance(iterator.args[0], ast.Constant)
                    and isinstance(iterator.args[0].value, int) and 0 <= iterator.args[0].value <= 3):
                raise ValueError('Skill loops must use literal range(0..3)')
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and abs(node.value) > 1000:
                raise ValueError('Oversized numeric constant')
            if isinstance(node.value, str) and len(node.value) > 1024:
                raise ValueError('Oversized string constant')
        if isinstance(node, ast.Pow):
            raise ValueError('Exponentiation is unsupported in bounded skills')
    return tree


def load_skill(path):
    source = Path(path).read_text(encoding='utf-8')
    tree = validate_skill_source(source)
    builtins = {'range': range, 'len': len, 'min': min, 'max': max, 'abs': abs,
                'float': float, 'int': int, 'bool': bool, 'tuple': tuple}
    namespace = {'__builtins__': builtins}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['run'], hashlib.sha256(source.encode('utf-8')).hexdigest()
