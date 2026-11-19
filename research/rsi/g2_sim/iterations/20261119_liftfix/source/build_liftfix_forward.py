from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'search_liftfix.py').read_text().replace("aid=sys.argv[1]","aid='basket_woven'")
s=s.replace('[(0,0,0,.003),(0,0,0,-.003),(0,.008,0,0),(0,-.008,0,0),(0,0,10,0),(0,0,-10,0),(0,0,0,0)]','[(.03,0,0,0),(.05,0,0,0),(.03,.008,0,0),(.06,.008,0,0)]')
s=s.replace("'liftfix_plans'","'liftfix_forward_plans'").replace("f'liftfix_search_{aid}","f'liftfix_forward_{aid}")
(R/'search_liftfix_forward.py').write_text(s)
