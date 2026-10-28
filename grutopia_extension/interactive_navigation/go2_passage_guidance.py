"""Keep the verified corridor entry as a subgoal after a sideways box push."""
import numpy as np


class PassageEntryGuide:
    def __init__(self,certificate,goal):
        self.entry=np.asarray(certificate['guide_entry'] if 'guide_entry' in certificate else certificate['corridor_start'],float)[:2].copy()
        self.exit=np.asarray(certificate['guide_exit'] if 'guide_exit' in certificate else certificate['corridor_end'],float)[:2].copy()
        self.goal=np.asarray(goal,float)[:2].copy()
        if not all(np.isfinite(x).all() for x in [self.entry,self.exit,self.goal]):
            raise ValueError('Passage guide coordinates must be finite')
        self.entry_reached=False

    @property
    def target(self):
        return self.goal if self.entry_reached else self.entry

    def advance(self,position,tolerance=.20):
        if not self.entry_reached and np.linalg.norm(np.asarray(position[:2])-self.entry)<tolerance:
            self.entry_reached=True
            return True
        return False

    def path(self,planner,position,boxes):
        if not self.entry_reached:return planner.path(position,self.entry,boxes)
        if float(np.dot(np.asarray(position[:2])-self.exit,self.exit-self.entry))>=0:
            return planner.path(position,self.goal,boxes)
        approach=planner.path(position,self.exit,boxes)
        after=planner.path(self.exit,self.goal,boxes)
        return [*approach,*after] if approach and after else []
