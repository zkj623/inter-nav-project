from pathlib import Path
import argparse,json,hashlib,subprocess,html
import imageio_ffmpeg
W=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
source=W/'evaluations'/a.run/'summary.json';summary=json.loads(source.read_text())
out=W/'pi05_focus_20261125/reviews'/a.run;out.mkdir(parents=True,exist_ok=True)
rows=[];ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
for row in summary['results']:
 trial=Path(row['trial']);video=trial/'review.mp4';target=out/(row['asset']+'.mp4')
 result=json.loads((trial/'result.json').read_text()) if (trial/'result.json').exists() else {}
 record={k:row.get(k) for k in ['asset','success','failure','rejections','max_lift_m','min_hold_lift_m','max_hold_tilt_deg']}
 record['hold_relative_drift_m']=result.get('hold_relative_drift_m')
 record['four_finger_contact_fraction_diagnostic']=result.get('hold_four_finger_contact_fraction')
 record['policy']={k:row.get('policy',{}).get(k) for k in ['checkpoint','sha256','step','model','sampling_seed','inputs','execute_steps','expert_feedback','server_source_sha256']}
 if video.exists():
  if not target.exists():
   with (out/(row['asset']+'_ffmpeg.log')).open('w') as log:
    subprocess.run([ffmpeg,'-nostdin','-y','-i',str(video),'-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart','-threads','2','-an',str(target)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
  probe=imageio_ffmpeg.read_frames(str(target));metadata=next(probe);frames=sum(1 for _ in probe)
  assert frames>0
  record.update(video=target.name,video_sha256=digest(target),video_frames=frames,video_fps=metadata['fps'])
 rows.append(record)
report=dict(run=a.run,successes=summary['successes'],total=summary['total'],split=summary['dataset_split'],acceptance=summary.get('protocol','stable'),checkpoint=summary['checkpoint'],items=rows)
(out/'review.json').write_text(json.dumps(report,indent=2))
parts=['<!doctype html><meta charset="utf-8"><title>π0.5 closed-loop review</title><style>body{font:16px sans-serif;max-width:1000px;margin:30px auto;color:#182331}video{width:100%;max-height:580px;background:#111}section{margin:30px 0}pre{white-space:pre-wrap}</style>',f'<h1>π0.5: {html.escape(a.run)}</h1>',f'<p>{summary["successes"]}/{summary["total"]} passed; split: {html.escape(summary["dataset_split"])}. Language + head/left-wrist/right-wrist RGB + qpos. No expert takeover.</p>','<p>Terminal-height test: reach the source-demo final lift (within 5 mm), then continue learned feedback for 10 seconds. No frozen targets, expert takeover or new hold prompt. Base contact diagnostic only. Report floor clearance and physical hold separately from last-5-second height/drift/rotation thresholds.</p>']
for row in rows:
 parts += [f'<section><h2>{html.escape(row["asset"])} — {"PASS" if row["success"] else "FAIL"}</h2>',f'<pre>{html.escape(json.dumps({k:v for k,v in row.items() if k not in ["policy","video_sha256"]},indent=2))}</pre>']
 if row.get('video'):parts.append(f'<video controls preload="metadata" src="{html.escape(row["video"])}"></video>')
 parts.append('</section>')
(out/'index.html').write_text('\n'.join(parts));print('REVIEW_READY',out,flush=True)
