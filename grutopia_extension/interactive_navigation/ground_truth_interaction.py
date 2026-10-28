"""Explicit layout-based interaction eligibility; does not populate a map."""
import hashlib
import json
from pathlib import Path
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission


class GroundTruthInteractionRuntime:
    source='ground_truth_layout'

    def __init__(self,catalog,layout_path,output_dir):
        self.catalog=catalog;self.output=Path(output_dir);self.output.mkdir(parents=True,exist_ok=True)
        path=Path(layout_path);raw=path.read_bytes();layout=json.loads(raw)['objects']
        self.objects={}
        for item in layout:
            object_id=str(item['id'])
            if object_id in self.objects:raise ValueError('Duplicate ground-truth object ID')
            category=str(item.get('category','box'))
            push=('push' in item.get('interactions',[]) and item.get('interactable',True)
                  and 'push_aside' in catalog.categories.get(category,()))
            self.objects[object_id]={'category':category,'skills':['push_aside'] if push else []}
        manifest={'version':1,'source':self.source,'layout_path':str(path.resolve()),
                  'layout_sha256':hashlib.sha256(raw).hexdigest(),'objects':self.objects,
                  'scope':'object categories and interaction eligibility only; no map initialization'}
        self.reference=self.output/'interaction_ground_truth.json'
        self.reference.write_text(json.dumps(manifest,indent=2,sort_keys=True))
        self.digest=hashlib.sha256(self.reference.read_bytes()).hexdigest()
        self.task_id=None

    def reset(self,task_id):
        if Path(str(task_id)).name!=str(task_id) or str(task_id) in ['.','..']:
            raise ValueError('Task ID must be a single path component')
        self.task_id=str(task_id);self.directory=self.output/self.task_id/'interaction'
        self.directory.mkdir(parents=True,exist_ok=True)

    def permission(self,object_id,skill_name,now):
        obj=self.objects.get(object_id)
        if obj is None or skill_name not in obj['skills']:return None
        return InteractionPermission(object_id,obj['category'],skill_name,(),(),float(now),
            source=self.source,source_reference=str(self.reference),source_digest=self.digest)

    def eligible_ids(self,now):
        return frozenset(key for key in self.objects if self.permission(key,'push_aside',now))

    def snapshot(self,now):
        return {key:{'category':obj['category'],'confirmed':True,'frames':0,'latest_at':None,
                     'source':self.source,'interactable':bool(obj['skills'])} for key,obj in self.objects.items()}

    def save(self,now):
        data={'required':False,'semantic_source':self.source,'source_reference':str(self.reference),
              'source_digest':self.digest,'objects':self.snapshot(now),
              'eligible_ids':sorted(self.eligible_ids(now)),'interactable_categories':list(self.catalog.categories)}
        (self.directory/'known_objects.json').write_text(json.dumps(data,indent=2))
        return data

    def wants_frame(self,now,objects):return False
    def poll(self):return None
    def close(self):pass
