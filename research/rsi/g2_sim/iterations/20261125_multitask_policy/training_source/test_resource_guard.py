import ast, copy, unittest
from pathlib import Path

source=ast.parse(Path(__file__).with_name('resume_guarded_eval.py').read_text())
guard=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='check')
class ResourceGuardTests(unittest.TestCase):
 def run_case(self, overrides, count=3):
  sample=dict(available_gib=100,swap_io_mib_s=0,iowait_percent=0,free_root_gib=100,free_data_gib=1000,gpus=[[0,10000,50],[1,10000,50]])
  sample.update(overrides)
  import json
  env={'snapshot':lambda:copy.deepcopy(sample),'strikes':0,'json':json}
  exec(compile(ast.Module(body=[guard],type_ignores=[]),'<guard>','exec'),env)
  for _ in range(count):env['check']()
 def test_healthy(self):self.run_case({})
 def test_severe_memory(self):
  with self.assertRaises(RuntimeError):self.run_case({'available_gib':15},1)
 def test_transient_pressure(self):self.run_case({'iowait_percent':40},2)
 def test_sustained_pressure(self):
  for change in [dict(available_gib=31),dict(swap_io_mib_s=21),dict(iowait_percent=30),dict(free_root_gib=29),dict(free_data_gib=99),dict(gpus=[[0,1000,50]]),dict(gpus=[[0,10000,86]])]:
   with self.subTest(change=change),self.assertRaises(RuntimeError):self.run_case(change)
if __name__=='__main__':unittest.main()
