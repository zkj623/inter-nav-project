import sys,json,platform,importlib.metadata as md
from pathlib import Path
report={'python':sys.version,'platform':platform.platform(),'packages':{}}
for name in ('numpy','scipy','opencv-python','matplotlib','usd-core','isaacsim'):
 try:report['packages'][name]=md.version(name)
 except md.PackageNotFoundError:report['packages'][name]='not in package metadata'
Path(sys.argv[1]).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
