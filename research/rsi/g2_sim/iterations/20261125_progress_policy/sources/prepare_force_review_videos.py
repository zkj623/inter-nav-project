from pathlib import Path
import os,json,subprocess,hashlib
import cv2,imageio_ffmpeg
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';O=W/'review_exports';O.mkdir(exist_ok=True)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10)
results=[]
for asset in ['box_white','bin_lidded','bin_pedal','basket_rectangular']:
 T=R/'policy_trials/act_force_v1_8000/placements/p0001'/asset/'candidate0_repeat1';source=T/'review.mp4';dest=T/'review_h264.mp4'
 if not dest.exists():
  with (O/(asset+'_transcode.log')).open('w') as log:
   subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-i',str(source),'-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-an','-threads','2',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 stats=[]
 for path in [source,dest]:
  cap=cv2.VideoCapture(str(path));assert cap.isOpened();frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);fourcc=int(cap.get(cv2.CAP_PROP_FOURCC));codec=''.join(chr((fourcc>>(8*i))&255) for i in range(4));ok,frame=cap.read();cap.release();assert ok and frames>60 and frame.std()>1;stats.append(dict(frames=frames,fps=fps,codec=codec))
 assert stats[0]['frames']==stats[1]['frames'] and abs(stats[0]['fps']-stats[1]['fps'])<1e-6
 assert stats[1]['codec'].lower() in ['avc1','h264']
 results.append(dict(asset=asset,original=stats[0],browser=stats[1],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),browser_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),browser_path=str(dest)))
 print('BROWSER_VIDEO_VERIFIED',asset,flush=True)
(W/'force_video_audit.json').write_text(json.dumps(results,indent=2))
