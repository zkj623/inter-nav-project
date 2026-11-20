from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'symhome_v4_bin_round_lidded_trial.py').read_text().replace("ROOT/'symhome_v4_trials'","ROOT/'symhome_v4staged_trials'")
source=(R/'symhome_v4_case_handheld_trial.py').read_text();a=source.index("    squeeze = {'l': 0., 'r': 0.}");b=source.index('    def four_contact(',a)
post=source[a:b].replace('    def retarget(', '    def post_lift_retarget(')
s=s.replace('    def retarget(', '    def preload_retarget(')
needle='    def four_contact(';pos=s.index(needle)
wrapper="""    use_post_lift=False
    def retarget(nominal,command,extra=0.):
        return post_lift_retarget(nominal,command,extra) if use_post_lift else preload_retarget(nominal,command,extra)
"""
s=s[:pos]+post+wrapper+s[pos:]
needle='            last=closed';assert s.count(needle)==1;s=s.replace(needle,'            use_post_lift=True\n'+needle)
s=s.replace("    result['contact_definition']=","    result['feedback_controller']='original preload control/gates; after verified preload, case-style tilt and continuous force balance target28N/side; all original four-finger settling and hold criteria retained'\n    result['contact_definition']=")
ast.parse(s);(R/'symhome_v4staged_bin_round_lidded_trial.py').write_text(s)
