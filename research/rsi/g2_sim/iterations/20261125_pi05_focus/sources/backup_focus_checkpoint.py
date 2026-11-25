from pathlib import Path
import sys,shutil,hashlib,json
src=Path(sys.argv[1]).resolve();dst=Path(sys.argv[2]).resolve()
assert src.is_relative_to(Path('/tmp/zkj_g2_20261125_checkpoints'))
assert dst.is_relative_to(Path('/data20t/embodied/zkj/inter-nav-results/g2-policy-overnight-20261124/checkpoint_backups/pi05_focus'))
assert (src/'_CHECKPOINT_METADATA').exists()
if not dst.exists():shutil.copytree(src,dst)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
checks=[]
for p in sorted(src.rglob('*')):
 if p.is_file():
  q=dst/p.relative_to(src);assert q.is_file() and p.stat().st_size==q.stat().st_size
  a=digest(p);assert a==digest(q);checks.append(dict(path=str(p.relative_to(src)),sha256=a))
(dst/'BACKUP_VERIFIED.json').write_text(json.dumps(dict(source=str(src),files=checks),indent=2));print('BACKUP_VERIFIED',dst)
