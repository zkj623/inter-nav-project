from pathlib import Path
import subprocess
R=Path('/data20t/embodied/zkj/inter-nav-results');repo=R/'g2-publish-20261118';checks=[R/'AGENTS.md',repo/'AGENTS.md',repo/'research/AGENTS.md',repo/'research/rsi/AGENTS.md',repo/'research/rsi/g2_sim/AGENTS.md',repo/'research/rsi/g2_sim/iterations/AGENTS.md']
for f in checks:
 if f.exists():print('INSTRUCTIONS',f,'\n'+f.read_text())
print('TRACKED_AGENT_FILES',[n for n in subprocess.check_output(['git','-C',str(repo),'ls-files'],text=True).splitlines() if n.endswith('AGENTS.md')]);print('GIT_STATUS',subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True))
