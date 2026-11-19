from pathlib import Path
import json,subprocess,sys
R=Path(__file__).resolve().parent
s=(R/'audit_fixedpose.py').read_text().replace("f'fixedpose_screen_{pid}.json'","f'fixedpose_search_{pid}.json'").replace("f'fixedpose_{pid}_validated_plans'","f'fixedpose_selected{pid}_validated_plans'").replace("f'fixedpose_{pid}_{kind}_audit.py'","f'fixedpose_selected{pid}_{kind}_audit.py'").replace("f'fixedpose_{pid}_audit_summary.json'","f'fixedpose_selected{pid}_audit_summary.json'")
(R/'audit_selected_fixedpose.py').write_text(s)
subprocess.run([sys.executable,str(R/'audit_selected_fixedpose.py'),'3'],check=True)
