"""Monotonic progress on a supplied route, with checked forward connections."""
import numpy as np

def trim_initial_attachment(start,path,clear,maximum_prefix_m=2.,maximum_chord_m=1.5,minimum_saving_m=.10):
    """Bypass a short backwards skeleton attachment, retaining the corridor route.

    The supplied checker uses the unchanged roadmap mask, including unknown
    cells. This changes only the initial attachment, not later skeleton bends;
    execution still performs its original full-body and braking checks.
    """
    if len(path)<2:return path,0
    start=np.asarray(start[:2],float);previous=start;arc=0.;chosen=0
    for i,point in enumerate(path):
        xy=np.asarray(point[:2],float);arc+=float(np.linalg.norm(xy-previous));previous=xy
        if arc>maximum_prefix_m+1e-9:break
        chord=float(np.linalg.norm(xy-start))
        if i and chord<=maximum_chord_m and arc-chord>=minimum_saving_m and clear(start,xy):chosen=i
    return path[chosen:],chosen

class RouteProgress:
    def __init__(self,start,path):
        self.points=np.asarray([start[:2],*[p[:2] for p in path]],float)
        self.lengths=np.linalg.norm(np.diff(self.points,axis=0),axis=1)
        self.arc=np.r_[0.,np.cumsum(self.lengths)];self.progress=0.

    def point(self,distance):
        distance=float(np.clip(distance,0.,self.arc[-1]))
        i=min(len(self.lengths)-1,max(0,int(np.searchsorted(self.arc,distance,side='right')-1)))
        fraction=(distance-self.arc[i])/max(self.lengths[i],1e-12)
        return self.points[i]+fraction*(self.points[i+1]-self.points[i])

    def advance(self,position,index,clear,lookahead=.55):
        position=np.asarray(position[:2],float);candidates=[]
        # The connection can skip a passed corner, not jump to a remote branch.
        bound=max(self.progress,self.arc[min(index,len(self.lengths)-1)])+max(1.,2*lookahead)
        for i in range(index,len(self.lengths)):
            if self.arc[i]>bound:break
            d=self.points[i+1]-self.points[i]
            fraction=float(np.clip(np.dot(position-self.points[i],d)/max(self.lengths[i]**2,1e-12),0.,1.))
            at=self.points[i]+fraction*d;s=self.arc[i]+fraction*self.lengths[i]
            error=float(np.linalg.norm(at-position))
            if s+1e-8<self.progress or error>.28 or s>bound:continue
            if not clear(position,at) or not clear(position,self.points[i+1]):continue
            candidates.append((error,s,i))
        if candidates:
            closest=min(row[0] for row in candidates)
            _,s,i=max((row for row in candidates if row[0]<=closest+.025),key=lambda row:row[1])
            self.progress=max(self.progress,float(s));index=max(index,i)
        return index

    def target(self,position,index,lookahead):
        i=min(index,len(self.lengths)-1);d=self.points[i+1]-self.points[i]
        fraction=float(np.clip(np.dot(np.asarray(position[:2])-self.points[i],d)/max(self.lengths[i]**2,1e-12),0.,1.))
        s=max(self.progress,self.arc[i]+fraction*self.lengths[i])
        return self.point(s+lookahead)
