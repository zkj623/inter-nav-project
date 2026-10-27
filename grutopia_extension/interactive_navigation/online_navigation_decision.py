"""Goal-directed frontiers and box decisions using only observed map cells."""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from grutopia_extension.interactive_navigation.box_push_navigation import BoxDecision
from grutopia_extension.interactive_navigation.semantic_voronoi import SemanticVoronoiGraph
from grutopia_extension.interactive_navigation.exploration_frontiers import local_unknown


class OnlineGoalSelector:
    def __init__(self):
        self.visited={};self.last_frontier=None;self.selection_count=0

    def reached_frontier(self,xy,observed_count):
        self.visited[self.key(xy)]=int(observed_count)

    @staticmethod
    def key(xy):return tuple(np.rint(np.asarray(xy[:2])/.40).astype(int))

    def decide(self,layers,start,goal):
        p=layers.planner;self.selection_count+=1
        with layers.global_planning() as boxes:
            # Unknown cells stay blocked even in the box-free hypothesis.
            optimistic=p.path(start,goal,[])
            if optimistic:
                decision=p.decide(start,goal,boxes)
                if decision.kind!='failed':
                    if decision.kind=='push':
                        box=next(b for b in boxes if b.id==decision.push.box_id)
                        passage=dict(p.active_passages[box.id])
                        passage['source']='observed_box_free_route'
                        axis=np.asarray(passage['axis'],float);axis/=np.linalg.norm(axis)
                        # Orient the crossing toward the original task goal.
                        if np.dot(np.asarray(goal)-passage['center'],axis)<0:axis=-axis
                        passage['axis']=axis.tolist()
                        passage['boundary']=float(np.dot(passage['center'],axis))
                        p.passages[box.id]=passage;p.active_passages[box.id]=passage
                    return decision,tuple(goal)
            grid=layers.frontier_grid();graph=SemanticVoronoiGraph(grid);graph.update()
            snapshot=graph.cached_snapshot()
            blocked=p.blocked(boxes);safe=~blocked
            if not safe.any():return BoxDecision('wait','insufficient_observed_space'),None
            _,nearest=distance_transform_edt(~safe,return_indices=True)
            proposals={}
            for frontier in snapshot.frontiers:
                cells=frontier.cells
                for rc in cells[::max(1,len(cells)//16)]:
                    row,col=nearest[:,rc[0],rc[1]]
                    xy=np.asarray(grid.cell_to_world((int(row),int(col))))
                    if np.linalg.norm(xy-np.asarray(start))<.45:continue
                    key=self.key(xy)
                    if key in self.visited and int(layers.observed.sum())-self.visited[key]<30:continue
                    gain=local_unknown(grid,xy,radius=.8)*p.resolution**2
                    score=np.linalg.norm(xy-np.asarray(start))+1.25*np.linalg.norm(xy-np.asarray(goal))-.8*gain
                    proposals[key]=(float(score),xy,float(gain))
            candidates=sorted(proposals.values(),key=lambda entry:entry[0])
            choices=[]
            for _,xy,gain in candidates[:32]:
                path=p.path(start,xy,boxes)
                if not path:continue
                cost=p.length(start,path)+1.25*float(np.linalg.norm(xy-np.asarray(goal)))-.8*gain
                choices.append((cost,xy,gain,path))
                if len(choices)>=8:break
            if not choices:
                return BoxDecision('wait','no_reachable_observed_frontier',explanation={
                    'kind':'wait','reason':'no_reachable_observed_frontier','start':list(start),'goal':list(goal),
                    'observed_cells':int(layers.observed.sum())}),None
            _,xy,gain,path=min(choices,key=lambda entry:entry[0]);self.last_frontier=tuple(xy)
            details={'kind':'explore','reason':'goal_not_connected_in_observed_map',
                'start':list(map(float,start)),'goal':list(map(float,goal)),
                'frontier_goal':xy.tolist(),'before_path':path,'before_length_m':p.length(start,path),
                'information_gain_m2':gain,'candidate_count':len(choices),
                'map_source':'online_lidar_observed_cells_only',
                'goal_observed':bool(grid.observed[p.cell(goal)])}
            return BoxDecision('explore','goal_not_connected_in_observed_map',path=path,explanation=details),tuple(xy)
