from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'verify_liftfix.py').read_text().replace("R.glob('liftfix*_trials/**/result.json')","R.glob('liftpair*_trials/**/result.json')").replace("'liftfix_verified_results.json'","'liftpair_verified_results.json'")
(R/'verify_liftpair.py').write_text(s)
s=(R/'report_liftfix.py').read_text().replace("R.glob('liftfix*_trials/**/result.json')","R.glob('liftpair*_trials/**/result.json')")
(R/'report_liftpair.py').write_text(s)
s=(R/'diagnose_liftfix.py').read_text().replace("R.glob('liftfix*_trials/**/result.json')","R.glob('liftpair*_trials/**/result.json')")
(R/'diagnose_liftpair.py').write_text(s)
