"""Fail if online navigation attempts to read precomputed scene geometry."""
import builtins,hashlib,io,json,runpy,shutil,sys
from pathlib import Path

entry=sys.argv.pop(1)
output=Path(sys.argv[sys.argv.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
shutil.copy2(__file__,output/'launch_online_checked.py')
attempts=[];original_open=builtins.open;original_io_open=io.open
forbidden={'navigation_grid.npz','original_metadata.json','passages.json','object_dict.json'}
def persist():
    # SimulationApp.close may exit without unwinding Python finally blocks.
    # Record that the guard is armed now and persist every violation before
    # raising, rather than depending on interpreter shutdown callbacks.
    with original_open(output/'online_input_audit.json','w') as stream:
        json.dump({'guard_armed':True,'scene_prior_reads_forbidden':sorted(forbidden),
            'prior_read_attempts':attempts,'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},stream,indent=2)
persist()
def check(file):
    if isinstance(file,(str,bytes,Path)):
        name=Path(file.decode() if isinstance(file,bytes) else file).name
        if name in forbidden:
            attempts.append(str(file));persist();raise RuntimeError('Online mode cannot read scene-prior data: '+str(file))
def guarded_open(file,*args,**kwargs):
    check(file);return original_open(file,*args,**kwargs)
def guarded_io_open(file,*args,**kwargs):
    check(file);return original_io_open(file,*args,**kwargs)
builtins.open=guarded_open;io.open=guarded_io_open
sys.argv[0]=entry
try:runpy.run_path(entry,run_name='__main__')
finally:
    builtins.open=original_open;io.open=original_io_open
    persist()
