"""Verify a user-owned24GiB scope, record its real PID, then exec unchanged job argv."""
from pathlib import Path
import os,json,sys
out=Path(sys.argv[1]);args=sys.argv[2:]
assert out.is_dir() and args
line=next(s for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
group=Path('/sys/fs/cgroup')/line.split(':',2)[2].lstrip('/')
maximum=int((group/'memory.max').read_text());swap=int((group/'memory.swap.max').read_text())
assert maximum==24*1024**3 and swap==0
assert group.name.startswith('codex-go2-') and group.name.endswith('.scope')
identity={'pid':os.getpid(),'parent_pid':os.getppid(),'pgid':os.getpgrp(),'sid':os.getsid(0),
    'cgroup':str(group),'unit':group.name,'memory_max_bytes':maximum,'swap_max_bytes':swap,'executable':args[0]}
(out/'scope_identity.json').write_text(json.dumps(identity,indent=2))
os.execv(args[0],args)
