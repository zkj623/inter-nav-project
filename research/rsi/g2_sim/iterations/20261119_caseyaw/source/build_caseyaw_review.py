from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'verify_commonstart.py').read_text().replace("R.glob('commonstart*_trials/**/result.json')","list(R.glob('caseyaw_trials/**/result.json'))+list(R.glob('caseyaw_level_trials/**/result.json'))+list(R.glob('caseyaw_damped_trials/**/result.json'))").replace("'commonstart_verified_results.json'","'caseyaw_verified_results.json'")
s=s.replace("exec(compile(s,'physical_review','exec'))", "s=s.replace(\"if d['success']:\",\"if d['success'] or any(x['phase']=='hold' for x in samples):\").replace(\"bool(not hits\",\"bool(d['success'] and not hits\")\nexec(compile(s,'physical_review','exec'))")
(R/'verify_caseyaw.py').write_text(s)
