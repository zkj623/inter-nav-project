"""Vision-backed category admission for robot interaction skills."""
from dataclasses import dataclass,field
from pathlib import Path
import json,math


BOX_ALIASES={'box':'box','cardboard box':'box','carton':'box',
             'plastic storage box':'box','wooden crate':'box'}


@dataclass(frozen=True)
class RecognitionEvidence:
    object_id:str
    category:str
    frame_id:str
    observed_at:float
    detector_score:float
    model:str
    image_path:str
    source:str
    raw_response:str=''


@dataclass
class ObjectRecognition:
    category:str='unknown'
    evidence:list=field(default_factory=list)
    seen_frames:set=field(default_factory=set)
    latest_at:float=-math.inf


class RecognitionStore:
    def __init__(self,min_frames=2,min_detector_score=.30,max_age_s=60.,aliases=None):
        self.min_frames=min_frames;self.min_detector_score=min_detector_score
        self.max_age_s=max_age_s;self.objects={};self.aliases=dict(BOX_ALIASES if aliases is None else aliases)

    def observe(self,evidence):
        # Simulator IDs, metadata labels and detector search words alone
        # cannot provide a semantic interaction permission.
        if (evidence.source!='rgb_dino_qwen_vl' or not evidence.model or not evidence.image_path
                or not evidence.frame_id or not evidence.object_id or not math.isfinite(evidence.observed_at)
                or not math.isfinite(evidence.detector_score)
                or not self.min_detector_score<=evidence.detector_score<=1.):return False
        item=self.objects.setdefault(evidence.object_id,ObjectRecognition())
        if evidence.frame_id in item.seen_frames:return False
        item.seen_frames.add(evidence.frame_id)
        raw_category=evidence.category.strip().casefold()
        if raw_category=='unknown':return False
        category=self.aliases.get(raw_category,raw_category)
        if item.category!=category:item.category=category;item.evidence=[]
        item.evidence.append(evidence);item.latest_at=float(evidence.observed_at)
        item.evidence=item.evidence[-max(8,self.min_frames):]
        return True

    def confirmed(self,object_id,now):
        item=self.objects.get(object_id)
        if item is None or len(item.evidence)<self.min_frames or now-item.latest_at>self.max_age_s:return None
        return item

    def snapshot(self,now):
        return {key:{'category':item.category,'confirmed':self.confirmed(key,now) is not None,
                     'frames':len(item.evidence),'latest_at':item.latest_at if math.isfinite(item.latest_at) else None,
                     'source':'rgb_dino_qwen_vl'} for key,item in self.objects.items()}


@dataclass(frozen=True)
class InteractionPermission:
    object_id:str
    category:str
    skill_name:str
    evidence_frames:tuple
    classifier_models:tuple
    issued_at:float


class InteractionCatalog:
    def __init__(self,categories=None,metadata=None):
        self.categories=categories if categories is not None else {'box':('push_aside',)}
        self.categories={str(k):tuple(v) for k,v in self.categories.items()}
        if 'unknown' in self.categories:raise ValueError('Unknown objects cannot be interactable')
        metadata=metadata or {}
        context=('chair','cabinet','table','person','bag','door','unknown')
        labels=[];terms=[];self.label_aliases=dict(BOX_ALIASES)
        for category in self.categories:
            defaults=('cardboard box','plastic storage box','wooden crate') if category=='box' else (category,)
            names=metadata.get(category,{}).get('recognition_labels',defaults)
            if 'unknown' in names:raise ValueError('Unknown cannot be an interaction label alias')
            labels.extend(names);self.label_aliases.update({str(name).casefold():category for name in names})
            terms.extend(metadata.get(category,{}).get('detection_terms',('cardboard box','carton','box') if category=='box' else (category,)))
        self.classification_labels=tuple(dict.fromkeys([*labels,*context]))
        self.proposal_query=' . '.join(dict.fromkeys([*terms,*context[:-1]]))+' .'

    @classmethod
    def load(cls,path):
        payload=json.loads(Path(path).read_text())
        if payload.get('version')!=1:raise ValueError('Unsupported interaction catalog version')
        return cls({key:value['skills'] for key,value in payload['categories'].items() if value.get('enabled',False)},payload['categories'])

    def permission(self,object_id,skill_name,recognitions,now):
        item=recognitions.confirmed(object_id,now)
        if item is None or skill_name not in self.categories.get(item.category,()):return None
        return InteractionPermission(object_id,item.category,skill_name,
            tuple(e.frame_id for e in item.evidence),tuple(sorted({e.model for e in item.evidence})),float(now))

    def eligible_ids(self,skill_name,recognitions,now):
        return frozenset(key for key in recognitions.objects if self.permission(key,skill_name,recognitions,now) is not None)
