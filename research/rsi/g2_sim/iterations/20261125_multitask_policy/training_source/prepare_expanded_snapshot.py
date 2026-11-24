from pathlib import Path
import json,time,subprocess,sys
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';deadline=time.monotonic()+1800
while True:
 wide=sum(len(json.loads(f.read_text())['accepted']) for f in (R/'wide_collection_state').glob('*.json'))
 if wide>=8 or time.monotonic()>deadline:break
 time.sleep(30)
assets=list(json.loads((R/'ground_lift_training_config.json').read_text())['validated_simulation_baselines'])
available=[asset for asset in assets if len(json.loads((R/'collection_state'/(asset+'.json')).read_text())['accepted'])>0]
subprocess.run([sys.executable,str(W/'prepare_overnight_data.py'),'--name','expanded12','--per-asset','8','--assets',*available],cwd=W,check=True)
print('EXPANDED_SNAPSHOT_READY',len(available),'classes',wide,'wide episodes available',flush=True)
