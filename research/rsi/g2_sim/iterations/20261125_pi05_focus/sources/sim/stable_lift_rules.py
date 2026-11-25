from pathlib import Path
def allowed_base_contact(event):
 return (Path(event['actor']).name=='base_link' and event['impulse']<=.02
         and event.get('minimum_separation',0)>=-.003)
