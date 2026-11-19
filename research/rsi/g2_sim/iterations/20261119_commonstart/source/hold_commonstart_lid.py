from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'commonstartslow_bin_lidded_trial.py').read_text().replace("ROOT/'commonstart_slow_trials'","ROOT/'commonstart_hold_trials'")
s=s.replace("                segment(last,last,3.,'hold')","                hold_start=step\n                while step-hold_start<360:\n                    corrected=retarget(q,last)\n                    segment(last,corrected,min(.1,(360-(step-hold_start))/120.),'hold');last=corrected")
ast.parse(s);(R/'commonstarthold_bin_lidded_trial.py').write_text(s)
