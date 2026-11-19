from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'search_marginpose_precise.py').read_text()
s=s.replace('for dx in [0.,-.04,.04,-.08,.08,-.12,.12]:','for dx in sorted([float(x) for x in np.arange(-.12,.121,.01)],key=abs):')
s=s.replace('marginpose_precise_search','marginpose_fine_search');ast.parse(s);(R/'search_marginpose_fine.py').write_text(s)
s=(R/'audit_marginpose_precise.py').read_text().replace('marginpose_precise','marginpose_fine')
(R/'audit_marginpose_fine.py').write_text(s)
