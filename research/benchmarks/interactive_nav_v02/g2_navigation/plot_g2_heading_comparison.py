"""Compare measured body headings along two physical navigation trajectories."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.font_manager import FontProperties
p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('current');p.add_argument('out');a=p.parse_args()
fontpath=Path('C:/Windows/Fonts/simsun.ttc');font=FontProperties(fname=str(fontpath)) if fontpath.exists() else FontProperties(family='DejaVu Serif')
sets=[json.loads(Path(a.baseline).read_text()),json.loads(Path(a.current).read_text())];titles=['Previous version: fixed heading','New version: forward preferred']
fig,axes=plt.subplots(1,2,figsize=(8.8,6.7),sharex=True,sharey=True)
allxy=np.vstack([np.array([r['position'][:2] for r in rows]) for rows in sets]);lo=allxy.min(0)-.5;hi=np.maximum(allxy.max(0)+.5,np.array([1.925,1.7]))
for ax,rows,title in zip(axes,sets,titles):
 xy=np.array([r['position'][:2] for r in rows]);yaw=np.array([r['yaw'] for r in rows]);distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
 ax.plot(xy[:,0],xy[:,1],color='black',linewidth=1.35,label='Measured trajectory')
 indices=np.unique(np.searchsorted(distance,np.arange(0,distance[-1],.55)).clip(max=len(rows)-1))
 for idx in indices:
  x,y=xy[idx];ax.arrow(x,y,.24*np.cos(yaw[idx]),.24*np.sin(yaw[idx]),head_width=.065,head_length=.08,length_includes_head=True,color='.35',linewidth=.7,zorder=4)
 ax.scatter(*xy[0],s=40,c='black',marker='o',label='Start',zorder=5);ax.scatter(1.325,1.1,s=105,facecolors='white',edgecolors='black',marker='*',label='Task goal',zorder=5)
 ax.add_patch(Circle((1.325,1.1),.5,fill=False,linestyle=':',edgecolor='.5',linewidth=1.))
 ax.set_title(title,fontproperties=font,fontsize=13,pad=10);ax.set_xlabel('World x (m)',fontproperties=font,fontsize=11);ax.set_aspect('equal');ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.grid(True,color='.89',linewidth=.6);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('World y (m)',fontproperties=font,fontsize=11);axes[1].legend(loc='lower right',frameon=False,fontsize=9)
fig.text(.5,.025,'Arrows indicate measured body heading; dotted circles show the unchanged 0.5 m goal region.',ha='center',fontproperties=font,fontsize=10)
fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(a.out,dpi=190,bbox_inches='tight');plt.close(fig)
