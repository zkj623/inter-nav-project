"""Remove per-asset planning/auditing concurrency from inactive collectors."""
from pathlib import Path
import hashlib,json,os,py_compile,time
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';names=['run_threecam_collection.py','run_wide_collection.py']
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  args=(p/'cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,ProcessLookupError):continue
 assert not any(str(R/n).encode() in args for n in names),'Do not modify an active collector'
B=W/'collector_serial_revision';B.mkdir(exist_ok=True);report={}
for name in names:
 p=R/name;before=p.read_bytes();s=before.decode('utf-8-sig')
 old='ThreadPoolExecutor(max_workers=len(assets))';assert s.count(old)==1
 assert not (B/(name+'.before')).exists()
 (B/(name+'.before')).write_bytes(before)
 s=s.replace(old,'ThreadPoolExecutor(max_workers=1)')
 p.write_text(s);py_compile.compile(str(p),doraise=True)
 report[name]=dict(before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),workers=1)
(B/'revision.json').write_text(json.dumps(dict(time=time.strftime('%FT%T'),files=report,scope='Planning, simulation and independent audit are serial across assets; no collection launched'),indent=2))
print('COLLECTOR_CONCURRENCY_FIXED_NO_JOB_STARTED')
