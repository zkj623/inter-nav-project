from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'search_marginpose_extended.py').read_text()
s=s.replace("rows=json.loads((R/'fixedpose_inventory.json').read_text())","rows=[x for x in json.loads((R/'fixedpose_inventory.json').read_text()) if x['asset']=='case_handheld']")
s=s.replace('if pe>.004 or re>.04:','if pe>.0005 or re>.005:')
s=s.replace("marginpose_search_{pid}","marginpose_precise_search_{pid}")
ast.parse(s);(R/'search_marginpose_precise.py').write_text(s)
s=(R/'audit_marginpose.py').read_text().replace('marginpose_selected','marginpose_precise').replace('marginpose_search','marginpose_precise_search')
(R/'audit_marginpose_precise.py').write_text(s)
