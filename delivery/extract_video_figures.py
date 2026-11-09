from pathlib import Path
import json,subprocess,hashlib
B=Path('E:/inter-nav-results/go2-textured-interaction-20261103');D=Path('E:/inter-nav-workspaces/go2-delivery-qv23/delivery');F=D/'figures';F.mkdir(exist_ok=True);manifest=[]
for i in range(1,11):
 t=f'T{i:02d}';v='qv13' if i<=5 else ('qv22' if i<=9 else 'qv23');g='existing-five' if i<=5 else 'interaction-five';run=B/g/'runs'/f'qualified-{v}-{t.lower()}-try1';folder=run/t/'video';rows=[json.loads(x) for x in (folder/'video_frames.jsonl').read_text().splitlines()]
 stage='skill_push' if i>=6 else 'explore_navigate';c=[r for r in rows if r['state']==t+' '+stage];start=c[len(c)//2];c=[r for r in rows if r['state']==t+' goal_settling'];end=c[-10]
 result=json.loads((run/t/'result.json').read_text())
 for kind,r in [('action',start),('goal',end)]:
  out=F/f'{t}-{v}-{kind}.png';video=folder/'combined.mp4';sec=r['frame_index']/25
  subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(sec),'-i',str(video),'-frames:v','1','-q:v','2','-y',str(out)],check=True)
  manifest.append({'task':t,'version':v,'kind':kind,'source_video':str(video),'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'frame_index':r['frame_index'],'video_time_s':sec,'state':r['state'],'image':out.relative_to(D).as_posix(),'recorded_goal_error_m':result['goal_error_m'],'recorded_success':result['success']})
(F/'frame-sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print('Extracted',len(manifest),'frames')
