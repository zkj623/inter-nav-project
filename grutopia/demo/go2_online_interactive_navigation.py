"""Go2 point navigation and side pushing from an initially unknown LiDAR map."""
import sys
from grutopia.demo.go2_side_clearance_navigation import main

if __name__=='__main__':
    if '--map-source' not in sys.argv:sys.argv.extend(['--map-source','online'])
    if '--path-controller' not in sys.argv:sys.argv.extend(['--path-controller','continuous'])
    raise SystemExit(main())
