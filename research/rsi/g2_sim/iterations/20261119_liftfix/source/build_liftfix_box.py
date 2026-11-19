from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'search_liftfix.py').read_text()
s=s.replace("aid=sys.argv[1]","aid='box_reinforced'")
s=s.replace('(.03,0,0,0),(.05,0,0,0),(.07,0,0,0),(.02,.025,0,0),(.04,.04,0,0),(0,.05,0,0),(.03,.05,10,0),(.03,.05,-10,0)','(.02,.025,0,0),(.04,.04,0,0),(.02,.04,0,0),(.03,.05,0,0),(0,.05,0,0)')
s=s.replace("target[:3,3]+=[dx,0,dz]","target[:3,3]+=[dx,.004 if side=='l' else -.004,dz]")
s=s.replace("stance_dx_m=dx,","stance_dx_m=dx,outset_each_hand_m=.004,")
s=s.replace("'liftfix_plans'","'liftfix_box_plans'").replace("f'liftfix_search_{aid}","f'liftfix_outset_{aid}")
(R/'search_liftfix_box.py').write_text(s)
