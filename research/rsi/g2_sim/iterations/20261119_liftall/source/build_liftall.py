from pathlib import Path
import ast,json
R=Path(__file__).resolve().parent
surfaces=['stool_rectangular','stool_wood_four_leg','table_side','basket_rectangular']
for label,src in [('general','marginpose_steady_trial.py'),('under','marginpose_supportsettle_trial.py')]:
 s=(R/src).read_text()
 import re
 s=re.sub(r"OUT=ROOT/[^\n]+",f"OUT=ROOT/'liftall_{label}_trials'/args.plan_dir/args.asset/f'candidate{{args.candidate}}_repeat{{args.repeat}}'",s,count=1)
 # Select the actual corrected collision asset used by the historical surface strategy.
 start=s.index("    objprim=stage.DefinePrim('/World/TestObject','Xform');")
 end=s.index('\n',start)
 s=s[:start]+f"    object_folder='strategy15_surface_assets' if args.asset in {surfaces!r} else 'object_assets'\n    result['object_collision_asset']=str(ROOT/object_folder/(args.asset+'.usdc'))\n    objprim=stage.DefinePrim('/World/TestObject','Xform');objprim.GetReferences().AddReference(str(ROOT/object_folder/(args.asset+'.usdc')))"+s[end:]
 s=s.replace('writer=None;samples=[];self_contacts=[];','writer=None;samples=[];self_contacts=[];nonhand_contacts=[];')
 marker="            if not (a.startswith('/robot/') and b.startswith('/robot/')):continue"
 assert marker in s
 s=s.replace(marker,"""            object_pair=(a.startswith('/robot/') and b.startswith('/World/TestObject')) or (b.startswith('/robot/') and a.startswith('/World/TestObject'))
            if object_pair:
                actor=a if a.startswith('/robot/') else b
                vals=data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]
                if not Path(actor).name.startswith('gripper_') and vals and len(nonhand_contacts)<500:
                    nonhand_contacts.append(dict(actor=actor,impulse=sum(float(np.linalg.norm(v.impulse)) for v in vals),minimum_separation=min(float(v.separation) for v in vals)))
"""+marker)
 s=s.replace("            samples.append(rec)","            if nonhand_contacts and max(x['impulse'] for x in nonhand_contacts)>.001:failure=failure or 'NONHAND_OBJECT_CONTACT'\n            if len(nonhand_contacts)>=500:failure=failure or 'NONHAND_CONTACT_BUFFER_FULL'\n            samples.append(rec)")
 s=s.replace("    (OUT/'self_contacts.json').write_text(json.dumps(self_contacts,indent=2))","    (OUT/'self_contacts.json').write_text(json.dumps(self_contacts,indent=2))\n    (OUT/'nonhand_contacts.json').write_text(json.dumps(nonhand_contacts,indent=2))")
 # Match grasp semantics: rim/handle pinch can bear on proximal finger surfaces.
 if label=='general':
  needle="    segment(qpre,qpre,2.,'settle')"
  helper="""    def four_contact(sample,threshold):
        forces=sample['finger_forces']
        return all(sum(v for n,v in forces.items() if (n==f'gripper_{side}_{finger}_link4' if plan.get('mode')=='bimanual_spread' else n.startswith(f'gripper_{side}_{finger}_')))>threshold for side in ('l','r') for finger in ('inner','outer'))
    result['contact_definition']='spread: four distal fingers; rim/handle pinch: four finger branches; >0.5 N each during hold'
"""
  assert needle in s;s=s.replace(needle,helper+needle)
  s=s.replace("ready=all(forces.get(f'gripper_{side}_{finger}_link4',0.)>1. for side in ('l','r') for finger in ('inner','outer')) and velocity<.015", "ready=four_contact(recent,1.) and velocity<.015")
  line="    four=lambda s:all(sum(v for n,v in s['finger_forces'].items() if n==f'gripper_{side}_{finger}_link4')>.5 for side in ('l','r') for finger in ('inner','outer'))"
  assert line in s;s=s.replace(line,'    four=lambda s:four_contact(s,.5)')
  s=s.replace("(plan.get('mode')!='bimanual_spread' or four_fraction>=.9)","four_fraction>=.9")
 ast.parse(s);(R/f'liftall_{label}_trial.py').write_text(s)
# All geometry audits use the same collision representation as runtime.
s=(R/'fixedpose_collision.py').read_text().replace("if aid=='stool_rectangular':",f"if aid in {surfaces!r}:")
(R/'liftall_collision.py').write_text(s)
s=(R/'audit_marginpose.py').read_text().replace("rows=json.loads((R/f'marginpose_search_{pid}.json').read_text())","rows=json.loads((R/f'marginpose_search_{pid}.json').read_text())")
s=s.replace("folder=f'marginpose_selected{pid}_validated_plans'","folder='liftall_validated_plans'")
s=s.replace("f'marginpose_selected{pid}_{kind}_audit.py'","f'liftall_{kind}_audit.py'")
s=s.replace("f'marginpose_selected{pid}_audit_summary.json'","'liftall_audit_summary.json'")
s=s.replace("script=R/f'liftall_{kind}_audit.py';",f"text=text.replace(\"aid=='stool_rectangular'\",\"aid in {surfaces!r}\")\n script=R/f'liftall_{{kind}}_audit.py';")
ast.parse(s);(R/'audit_liftall.py').write_text(s)
print('Built uniform fixed-body harness, mode-aware four-contact criteria and runtime nonhand contact logging')
