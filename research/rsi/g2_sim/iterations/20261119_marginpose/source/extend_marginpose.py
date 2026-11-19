from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'search_marginpose.py').read_text()
s=s.replace('pid=int(sys.argv[1]);',"profiles += [[-45,142,-82,0,0],[-40,142,-87,0,0],[-35,142,-92,0,0],[-35,140,-87,0,0]]\npid=int(sys.argv[1]);")
ast.parse(s);(R/'search_marginpose_extended.py').write_text(s)
