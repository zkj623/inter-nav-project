from pathlib import Path
import json,shutil
R=Path(__file__).resolve().parent;rows=json.loads((R/'symhome_verified_results.json').read_text());out=R/'symhome_review';(out/'videos').mkdir(exist_ok=True)
selected=[]
for a in sorted({r['asset'] for r in rows}):
 group=[r for r in rows if r['asset']==a]
 selected.append(next((r for r in group if r['verified_success']),next((r for r in group if r['result'].startswith('symhome_v4_trials/')),group[0])))
names={'basket_rectangular':'方篮','basket_woven':'编织篮','bin_lidded':'带盖桶','bin_pedal':'脚踏桶','bin_round_lidded':'圆带盖桶','bin_round_pedal':'圆脚踏桶','box_reinforced':'加固箱','box_white':'白箱','case_handheld':'手提硬箱','stool_rectangular':'长方凳','stool_wood_four_leg':'四脚木凳','table_side':'边桌'}
xml='<h2>各物体新版初姿完整视频</h2><p>以下均从相同默认关节目标连续进入相同新版初姿，再开始物体交互。优先展示通过独立复核的版本；未通过的物体明确标为待适配。失败比较与完整指标保留在代码归档中。</p>'
for r in selected:
 a=r['asset'];state='通过' if r['verified_success'] else '待适配';xml+=f'<h3>{names.get(a,a)}：{state}</h3>'
 if a=='box_white':xml+='<figure view-type="Preview"><source token="Dl07bnKVrogmqTx9FHncwsdPnbg" name="G2_默认站姿_对称初姿_白箱完整抓取.mp4"/></figure>'
 else:
  shutil.copy2(r['review.mp4']['server_path'],out/'videos'/(a+'.mp4'))
  xml+=f'<figure view-type="Preview"><source path="@./videos/{a}.mp4" name="G2_新版初姿_{names.get(a,a)}_{state}.mp4"/></figure>'
(out/'videos.xml').write_text(xml,encoding='utf-8');(out/'video_selection.json').write_text(json.dumps(selected,indent=2));print('VIDEOS_READY',len(selected))
