"""Add an explicit final-outcome caption without changing simulation frames."""
from pathlib import Path
import argparse,json,cv2
p=argparse.ArgumentParser();p.add_argument('trial',type=Path);p.add_argument('--name',required=True);a=p.parse_args()
d=json.loads((a.trial/'result.json').read_text());R=Path(__file__).resolve().parent
relative=str((a.trial/'result.json').resolve().relative_to(R))
verified=next(r for r in json.loads((R/'push_result_verification.json').read_text()) if r['result']==relative)
passed=verified['review_accepted'];cap=cv2.VideoCapture(str(a.trial/'review.mp4'));assert cap.isOpened()
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=cap.get(cv2.CAP_PROP_FPS)
out=a.trial/(a.name+'.mp4');writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height));assert writer.isOpened()
label='PASS' if passed else 'FAIL'
text=f"{label} | final push {100*d['push_forward_m']:.1f} cm"
if 'max_yaw_error_deg' in d and d.get('failure') in [None,'PUSH_STOP_CRITERIA_NOT_MET']:text+=f" | max yaw {d['max_yaw_error_deg']:.2f} deg"
if verified.get('max_lateral_drift_m') is not None:text+=f" | side {100*verified['max_lateral_drift_m']:.2f} cm"
text+=' | 1kg rigid slide'
while True:
 ok,frame=cap.read()
 if not ok:break
 cv2.rectangle(frame,(0,height-42),(width,height),(20,20,20),-1);cv2.putText(frame,text,(16,height-14),cv2.FONT_HERSHEY_SIMPLEX,.63,(120,255,140) if passed else (70,150,255),1,cv2.LINE_AA);writer.write(frame)
cap.release();writer.release();print(out)
