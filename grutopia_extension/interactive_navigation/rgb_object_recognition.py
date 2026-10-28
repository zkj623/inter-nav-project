"""RGB classification with RGB-D association; geometry never supplies the class."""
from dataclasses import dataclass
from pathlib import Path
import json,time
import cv2
import numpy as np

from grutopia_extension.interactive_navigation.interaction_catalog import RecognitionEvidence
from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    AgentVLMBackend,OpenVocabularyPerceptionConfig,OpenVocabularyPerception)


CLASS_LABELS=('box','cardboard box','plastic storage box','wooden crate','chair','cabinet',
              'table','person','bag','door','unknown')
PROPOSAL_QUERY='cardboard box . carton . box . chair . cabinet . table . person . bag .'


@dataclass(frozen=True)
class ObjectGeometry:
    object_id:str
    center:tuple
    dimensions:tuple


def associate_points(points,objects,margin=.08,min_fraction=.55):
    if not len(points):return None,0.
    points=np.asarray(points,float);choices=[]
    for obj in objects:
        delta=np.abs(points-np.asarray(obj.center))
        inside=np.all(delta<=np.asarray(obj.dimensions)/2+margin,axis=1)
        fraction=float(np.mean(inside))
        if fraction>=min_fraction:choices.append((fraction,obj.object_id))
    if not choices:return None,0.
    fraction,object_id=max(choices)
    return object_id,fraction



def emphasize_instance(image,mask):
    """Keep observed foreground RGB intact and reduce surrounding clutter."""
    image=np.asarray(image);mask=np.asarray(mask,bool)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or mask.shape!=image.shape[:2]:
        raise ValueError('Foreground emphasis requires matching uint8 RGB/BGR image and mask')
    muted=(image.astype(np.float32)*.15+192.*.85).astype(np.uint8)
    muted[mask]=image[mask]
    return muted


def context_crop(image,bbox,side=384):
    x1,y1,x2,y2=map(float,bbox);h,w=image.shape[:2]
    dx=.15*(x2-x1);dy=.15*(y2-y1)
    l=max(0,int(np.floor(x1-dx)));t=max(0,int(np.floor(y1-dy)))
    r=min(w,int(np.ceil(x2+dx)));b=min(h,int(np.ceil(y2+dy)))
    crop=image[t:b,l:r].copy()
    if crop.size==0:raise ValueError('Empty recognition crop')
    cv2.rectangle(crop,(max(0,int(x1)-l),max(0,int(y1)-t)),
                  (min(r-l-1,int(x2)-l),min(b-t-1,int(y2)-t)),(0,0,255),2)
    scale=side/max(crop.shape[:2])
    return cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))),
                      interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)


class RGBObjectRecognizer:
    def __init__(self,backend=None,min_score=.30,min_points=30,max_candidates=6,labels=CLASS_LABELS,proposal_query=PROPOSAL_QUERY):
        self.backend=backend or AgentVLMBackend(OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl'))
        self.min_score=min_score;self.min_points=min_points;self.max_candidates=max_candidates
        self.labels=tuple(labels);self.proposal_query=str(proposal_query)

    def check_ready(self):return self.backend.check_ready()

    def recognize(self,camera_data,objects,frame_id,observed_at,output_dir):
        rgba=np.asarray(camera_data['rgba']);depth=np.asarray(camera_data['depth'])
        points=np.asarray(camera_data['pointcloud_image'])
        if rgba.ndim!=3 or depth.shape!=rgba.shape[:2] or points.shape!=(*depth.shape,3):
            raise ValueError('Recognition requires synchronized RGB, depth and world-point images')
        image=cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGR if rgba.shape[2]==4 else cv2.COLOR_RGB2BGR)
        output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
        safe_id=''.join(c if c.isalnum() or c in '._-' else '_' for c in str(frame_id))
        frame_path=output/f'{safe_id}.jpg';cv2.imwrite(str(frame_path),image)
        started=time.monotonic();debug={'frame_id':str(frame_id),'observed_at':float(observed_at),
            'image_path':str(frame_path),'semantic_source':'rgb_dino_qwen_vl',
            'geometry_source':'lidar_visible_geometry_proposals','detections':[]}
        raw=self.backend.detect(image,self.proposal_query)
        proposals=[]
        for item in raw:
            label,score,bbox=OpenVocabularyPerception._parse_detection(item,image.shape[:2])
            if score>=self.min_score and (bbox[2]-bbox[0])*(bbox[3]-bbox[1])>=1024:
                proposals.append((score,label,bbox))
        evidence=[];claimed=set()
        for score,label,bbox in sorted(proposals,reverse=True)[:self.max_candidates]:
            mask=np.asarray(self.backend.segment(image,bbox),bool)
            if mask.shape!=depth.shape:raise ValueError('SAM mask must match captured image')
            valid=mask & np.isfinite(depth) & (depth>.05) & (depth<5.) & np.isfinite(points).all(axis=-1)
            world=points[valid]
            item={'detector_label':label,'detector_score':float(score),'bbox_xyxy':list(bbox),
                  'valid_depth_points':int(len(world))}
            debug['detections'].append(item)
            if len(world)<self.min_points:item['status']='insufficient_depth';continue
            sample=world[::max(1,len(world)//1024)]
            object_id,fraction=associate_points(sample,objects)
            item.update(object_id=object_id,association_fraction=fraction)
            if object_id is None:item['status']='unassociated_geometry';continue
            if object_id in claimed:item['status']='duplicate_object_this_frame';continue
            claimed.add(object_id)
            context=context_crop(image,bbox)
            context_path=output/f'{safe_id}_context{len(evidence)}.jpg';cv2.imwrite(str(context_path),context)
            crop=context_crop(emphasize_instance(image,mask),bbox)
            crop_path=output/f'{safe_id}_crop{len(evidence)}.jpg';cv2.imwrite(str(crop_path),crop)
            item.update(classification_input='sam_foreground_with_muted_context',context_crop_path=str(context_path))
            classified=self.backend.classify(crop,self.labels)
            category=classified.get('label');model=classified.get('model')
            if category not in self.labels or not isinstance(model,str) or not model:
                raise ValueError('Classifier must return an explicit supported label and model provenance')
            item.update(status='classified',category=category,classifier_model=model,
                        classifier_response=classified.get('raw_text',''),crop_path=str(crop_path))
            evidence.append(RecognitionEvidence(object_id,category,str(frame_id),float(observed_at),
                float(score),model,str(crop_path),'rgb_dino_qwen_vl',raw_response=classified.get('raw_text','')))
        debug['wall_seconds']=time.monotonic()-started
        (output/f'{safe_id}.json').write_text(json.dumps(debug,indent=2))
        return evidence,debug
