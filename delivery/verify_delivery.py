from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
failed=[]
for name,expected in manifest.items():
    p=root/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:failed.append(name)
print(f'Checked {len(manifest)} delivery files; failures: {len(failed)}')
for name in failed:print(name)
sys.exit(bool(failed))
