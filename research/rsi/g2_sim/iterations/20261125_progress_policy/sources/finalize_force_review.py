from pathlib import Path
import subprocess,json,html,datetime,hashlib,shutil
L=Path(__file__).resolve().parent;V=L/'policy_reviews';V.mkdir(exist_ok=True)
W='/data20t/embodied/zkj/inter-nav-results/g2-policy-overnight-20261124';R='/data20t/embodied/zkj/inter-nav-results/g2-act-threecam-20261124';H='zkj@162.105.195.166'
def get(remote,target):subprocess.run(['scp','-q',H+':'+remote,str(target)],check=True)
get(W+'/force_video_audit.json',V/'force_video_audit.json')
video_audit={r['asset']:r for r in json.loads((V/'force_video_audit.json').read_text())}
get(W+'/evaluations/act_force_v1_8000/summary.json',V/'act_force_v1_summary_raw.json')
summary=json.loads((V/'act_force_v1_summary_raw.json').read_text());assert summary['total']==4 and summary['successes']==2
items=[]
for row in summary['results']:
 asset=row['asset'];T=row['trial'];raw=V/(asset+'_result_raw.json');get(T+'/result.json',raw);d=json.loads(raw.read_text())
 for name in ['verification.json','force_input_audit.json']:
  get(T+'/'+name,V/('act_force_v1_'+asset+'_'+name))
 if row['success']:get(T+'/acceptance.json',V/('act_force_v1_'+asset+'_threecam_acceptance.json'))
 video=V/('act_force_v1_'+asset+'_h264.mp4')
 if not video.exists():get(T+'/review_h264.mp4',video)
 assert hashlib.sha256(video.read_bytes()).hexdigest()==video_audit[asset]['browser_sha256']
 policy=row['policy'];calls=policy.pop('inference_calls',[]);policy['inference_call_count']=len(calls)
 metrics={k:d.get(k) for k in ['min_hold_lift_m','max_lift_m','max_hold_tilt_deg','hold_four_finger_contact_fraction','hold_relative_drift_m','failure']}
 items.append(dict(asset=asset,success=row['success'],video=video.name,metrics=metrics,policy=policy,trial=T))
data=dict(created=datetime.datetime.now().isoformat(),run=summary['run'],successes=2,total=4,checkpoint=summary['checkpoint'],items=items,scope='Fixed training positions, time-conditioned ACT with simulated contact forces; no generalization or hardware claim')
(V/'act_force_v1_review_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
esc=html.escape;labels=dict(bin_lidded='带盖桶',basket_rectangular='矩形篮',box_white='白箱',bin_pedal='脚踏桶');reasons=dict(box_white='左侧内指没有接触；四指接触比例为 0。',bin_pedal='抬升约 3.6 cm 后发生底盘碰撞。')
parts=['''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G2 ACT 闭环验证</title><style>body{font:16px system-ui;max-width:1120px;margin:32px auto;padding:0 20px;color:#202124;line-height:1.65}h1{font-size:28px}.scope{background:#f2f5f8;padding:16px;border-radius:10px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.card{border:1px solid #ddd;border-radius:10px;padding:16px}video{width:100%}.ok{color:#147548}.bad{color:#9a4a00}table{border-collapse:collapse;width:100%;margin:24px 0}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}code{overflow-wrap:anywhere}@media(max-width:760px){.grid{grid-template-columns:1fr}} </style><h1>同一 ACT 模型：2 / 4 物体通过闭环验收</h1><p class="scope">同一检查点、同一执行设置，无专家接管。固定机器人身体和训练时出现过的物体位置。这是多物体过拟合验证，尚未完成 12 物体或随机位置泛化验证。</p><table><tr><th>物体</th><th>结果</th><th>保持高度</th><th>保持倾角</th><th>四指接触比例</th></tr>''']
order=sorted(items,key=lambda x:not x['success'])
for item in order:
 m=item['metrics'];height=f"{m['min_hold_lift_m']*100:.2f} cm" if m['min_hold_lift_m'] is not None else f"未保持；最高 {m['max_lift_m']*100:.2f} cm";tilt=f"{m['max_hold_tilt_deg']:.2f}°" if m['max_hold_tilt_deg'] is not None else '—';fraction=f"{m['hold_four_finger_contact_fraction']*100:.0f}%" if m['hold_four_finger_contact_fraction'] is not None else '未进入保持'
 parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [labels[item['asset']],'通过' if item['success'] else '未通过',height,tilt,fraction])+'</tr>')
parts.append('</table><div class="grid">')
for item in order:
 asset=item['asset'];text='通过物理、接触输入时序及三路相机数据验收。' if item['success'] else reasons[asset]
 parts.append('<article class="card"><h2 class="'+('ok' if item['success'] else 'bad')+'">'+labels[asset]+(' · 通过' if item['success'] else ' · 未通过')+'</h2><p>'+text+'</p><video controls preload="metadata" src="'+esc(item['video'])+'"></video></article>')
parts.append('''</div><details><summary>模型输入、输出和验证范围</summary><p>输入：头部 RGB、左右腕部 RGB、16 维双臂/夹爪关节状态、任务 ID、经过时间，以及四个手指的仿真接触力。接触力仅使用最近已发生的测量，不含未来信息；未确认真机传感器支持。</p><p>输出：50 步动作块；每次执行 2 步后重新观测。目标控制频率 20 Hz，重新规划频率 10 Hz。身体固定，使用原有碰撞、几何接触和关节约束。</p><p>每个物体本轮各测试一次。通过要求包括至少 15 cm 抬升、稳定保持和四指接触比例不低于 90%，本轮未放宽。π0.5 先前配置为 0/4，本轮未重训；不能据此判定算法优劣。</p><p>检查点：<code>'''+esc(data['checkpoint'])+'</code></p><p>SHA-256：<code>'+esc(items[0]['policy']['sha256'])+'</code></p></details><p>更新：'+esc(data['created'])+'</p></html>')
(V/'act_force_v1_review.html').write_text('\n'.join(parts),encoding='utf-8')
print('REVIEW_READY',V/'act_force_v1_review.html')
