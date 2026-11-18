"""Generate a scene comparison figure and a local HTML inspection index."""
import argparse,json,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();fig,axs=plt.subplots(2,5,figsize=(22,10));cards=[]
for ax,p in zip(axs.flat,sorted(a.out.glob('S*/episodes.json'))):
 d=json.loads(p.read_text());g=np.load(p.parent/'evaluation_grid.npz');origin=g['origin'];r=.1;shape=g['floor'].shape;extent=[origin[0],origin[0]+shape[1]*r,origin[1],origin[1]+shape[0]*r]
 grid=np.ones((*shape,3));grid[g['floor']>0]=.92;grid[g['blocked']>0]=.28
 ax.imshow(grid,origin='lower',extent=extent)
 for t in d['tasks'][::2]:
  xy=np.array(t['reference_route_xy_m']);ax.plot(xy[:,0],xy[:,1],color='#407090',linewidth=.8,alpha=.7);ax.plot(*xy[0],'o',color='#237b45',ms=3);ax.plot(*xy[-1],'x',color='#b32929',ms=4)
 for o in d['objects']:
  x,y,_=o['position_m'];w,h=o['dimensions_m'][:2];ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,color='#dc9230'));ax.annotate(o['instance_id'],(x,y),xytext=(3,4),textcoords='offset points',fontsize=6)
 ax.set_title(f'{d["id"]} | {d["floor_area_m2"]:.1f} m²\n{d["scene_id"][:-4]}',fontsize=9);ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.tick_params(labelsize=7)
 single,one=plt.subplots(figsize=(8,7));one.imshow(grid,origin='lower',extent=extent)
 for t in d['tasks'][::2]:
  xy=np.array(t['reference_route_xy_m']);one.plot(xy[:,0],xy[:,1],lw=1,label=t['id']);one.plot(*xy[0],'go');one.plot(*xy[-1],'rx')
 for o in d['objects']:
  x,y,_=o['position_m'];w,h=o['dimensions_m'][:2];one.add_patch(Rectangle((x-w/2,y-h/2),w,h,color='#dc9230'));one.text(x,y,o['instance_id'],fontsize=8)
 one.set_title(f'{d["id"]} | {d["floor_area_m2"]:.2f} m²');one.set_aspect('equal');one.legend(fontsize=7);single.tight_layout();single.savefig(p.parent/'layout.png',dpi=160);plt.close(single)
 rows=''.join(f'<tr><td>{o["instance_id"]}</td><td>{html.escape(o["name"])}</td><td>{o["asset_id"]}</td></tr>' for o in d['objects'])
 cards.append(f'<section><h2>{d["id"]} · {d["floor_area_m2"]:.2f} m²</h2><p>{d["scene_id"]}</p><img src="{d["id"]}/layout.png"><table>{rows}</table><p><a href="{d["id"]}/episodes.json">任务配置</a> · 使用 materialize.py 生成 USD 场景</p></section>')
fig.suptitle('GRScenes Interactive Navigation v0.1 | 10 scenes · 15 object types · 60 tasks\nOrange: objects; green: start; red: goal; lines: reference routes (objects removed)',fontsize=14);fig.subplots_adjust(left=.04,right=.99,bottom=.07,top=.86,hspace=.38,wspace=.25);fig.savefig(a.out/'overview.png',dpi=160)
(a.out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Benchmark v0.1</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;color:#222}main{display:grid;grid-template-columns:1fr 1fr;gap:24px}section{border:1px solid #ddd;padding:20px}img{width:100%}td{padding:6px;border-bottom:1px solid #ddd}p{overflow-wrap:anywhere}</style><h1>交互式导航 Benchmark v0.1</h1><p>10 个家庭场景 · 15 种物体 · 30 个实例 · 60 个任务。几何筛选与物理检查请分别查看验证报告；机器人闭环评测待完成。</p><img src="overview.png"><main>'+''.join(cards)+'</main>',encoding='utf-8')
