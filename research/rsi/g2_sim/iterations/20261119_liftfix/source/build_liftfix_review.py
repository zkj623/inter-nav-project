from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'verify_liftall.py').read_text().replace("R.glob('liftall_*_trials/**/result.json')","R.glob('liftfix*_trials/**/result.json')").replace("'liftall_verified_results.json'","'liftfix_verified_results.json'")
(R/'verify_liftfix.py').write_text(s)
s=(R/'diagnose_liftall.py').read_text().replace("R.glob('liftall_*_trials/**/result.json')","R.glob('liftfix*_trials/**/result.json')").replace("print(d['asset'],d.get('failure'),flush=True)","print(str(p.relative_to(R)),d.get('failure'),flush=True)")
(R/'diagnose_liftfix.py').write_text(s)
s=(R/'report_liftall.py').read_text();s=s[:s.index("if (R/'liftall_batch_events.json')") ] if "if (R/'liftall_batch_events.json')" in s else s
s=s.replace("R.glob('liftall*_trials/**/result.json')","R.glob('liftfix*_trials/**/result.json')").replace("R.glob('liftall_*_trials/**/result.json')","R.glob('liftfix*_trials/**/result.json')")
if "p=R/'liftall_batch_events.json'" in s:s=s[:s.index("p=R/'liftall_batch_events.json'")]
(R/'report_liftfix.py').write_text(s)
