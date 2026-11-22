"""Persist completion monitoring independently of the interactive SSH session.

Never marks an incomplete collection successful, never modifies physical settings.
"""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent
def save(state,**kw):
    p=R/'completion_watch.json'
    tmp=p.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(dict(state=state,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2))
    tmp.replace(p)
def processes():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            if (p/'cwd').resolve()!=R:continue
            args=(p/'cmdline').read_bytes().split(b'\0')
            if any(x.endswith(b'_trial.py') or x.endswith(b'run_collection_batch.py') or x.endswith(b'run_collection_extra.py') for x in args):found.append(int(p.name))
        except (FileNotFoundError,PermissionError):pass
    return found
while True:
    try:
        progress=json.loads((R/'collection_progress.json').read_text())
        live=processes()
        full=all(n>=20 for n in progress['counts'].values()) and len(progress['counts'])==12
        save('collecting' if live else 'checking',counts=progress['counts'],verified=progress['total_verified'],live_pids=live)
        if not live:
            if not full:
                save('incomplete_requires_iteration',counts=progress['counts'],verified=progress['total_verified'],reason='Collectors ended before all 12 classes reached 20; dataset is not frozen')
                raise SystemExit(2)
            for script in ['finalize_collection.py','package_collection.py']:
                with (R/(script+'.log')).open('w') as out:
                    subprocess.run([sys.executable,str(R/script)],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
            save('complete_frozen',summary=str(R/'dataset/frozen_summary.json'),manifest=str(R/'dataset/frozen_manifest.jsonl'),archive='g2-publish-20261118/research/rsi/g2_sim/iterations/20261122_collection',note='Final archive generated locally; review and push the completed snapshot separately')
            break
    except (FileNotFoundError,json.JSONDecodeError):pass
    except Exception as e:
        save('completion_error',error=repr(e));raise
    time.sleep(30)
