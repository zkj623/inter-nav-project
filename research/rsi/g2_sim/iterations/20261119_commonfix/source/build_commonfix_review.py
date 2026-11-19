from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'verify_commonstart.py').read_text().replace("R.glob('commonstart*_trials/**/result.json')","R.glob('commonfix*_trials/**/result.json')").replace("'commonstart_verified_results.json'","'commonfix_verified_results.json'")
s += "\nrows=json.loads((R/'commonfix_verified_results.json').read_text())\nfor r in rows:\n d=json.loads((R/r['result']).read_text())\n if d.get('precision_collision_experiment') and not d.get('precision_material_preserved'):\n  r['verified_success']=False;r['diagnostic_only_reason']='precision prototype missing explicit original friction material binding; superseded by accurate variant'\n(R/'commonfix_verified_results.json').write_text(json.dumps(rows,indent=2))\n"
(R/'verify_commonfix.py').write_text(s)
s=(R/'report_commonstart.py').read_text().replace("R.glob('commonstart*_trials/**/result.json')","R.glob('commonfix*_trials/**/result.json')")
(R/'report_commonfix.py').write_text(s)
