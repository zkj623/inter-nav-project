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

    def consume_execution_failure(self,xy,observed_count,tracking):
        """Reject an attempted frontier before a new tracker hides its failure.

        Reuse the existing observed-information retry policy. This only
        removes an unproductive exploration destination; it neither marks
        unknown space free nor rejects the original navigation goal.
        """
        if not tracking.get('needs_replan',False):return None
        staging_failed=tracking.get('staging_phase')=='staging_failed'
        blocked_ticks=int(tracking.get('blocked_ticks',0))
        if not staging_failed and blocked_ticks<=30:return None
        self.reached_frontier(xy,observed_count)
        return {'frontier':list(map(float,xy[:2])),
            'reason':('staging_failed' if staging_failed else 'no_measured_translation'
                      if tracking.get('execution_stalled') else 'controller_motion_blocked'),
            'controller':tracking.get('controller'),'blocked_ticks':blocked_ticks,
            'observed_count':int(observed_count)}

    @staticmethod
    def key(xy):return tuple(np.rint(np.asarray(xy[:2])/.40).astype(int))

    def task_decision(self,layers,start,goal):
        """A geometric box-free path alone is not an executable task plan."""
        p=layers.planner
        with layers.global_planning() as boxes:
            # Unknown cells stay blocked even in the box-free hypothesis.
            optimistic=p.path(start,goal,[])
            if optimistic:
                decision=p.decide(start,goal,boxes)
                if decision.kind!='failed':
                    decision.explanation.update(map_source='online_lidar_observed_cells_only',
                        requested_goal=list(map(float,goal)),goal_observed=True)
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
                    return decision
        return None

    def decide(self,layers,start,goal):
        p=layers.planner;self.selection_count+=1
        task_decision=self.task_decision(layers,start,goal)
        if task_decision is not None:return task_decision,tuple(goal)
        with layers.global_planning() as boxes:
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
                    'kind':'wait','reason':'no_reachable_observed_frontier','start':list(map(float,start)),'goal':list(map(float,goal)),
                    'observed_cells':int(layers.observed.sum())}),None
            _,xy,gain,path=min(choices,key=lambda entry:entry[0]);self.last_frontier=tuple(xy)
            details={'kind':'explore','reason':'goal_not_connected_in_observed_map',
                'start':list(map(float,start)),'goal':list(map(float,goal)),
                'frontier_goal':xy.tolist(),'before_path':path,'before_length_m':p.length(start,path),
                'information_gain_m2':gain,'candidate_count':len(choices),
                'map_source':'online_lidar_observed_cells_only',
                'goal_observed':bool(grid.observed[p.cell(goal)])}
            return BoxDecision('explore','goal_not_connected_in_observed_map',path=path,explanation=details),tuple(xy)
