"""Asynchronous visual recognition, with task epochs and distinct-frame evidence."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
from grutopia_extension.interactive_navigation.interaction_catalog import RecognitionStore


class InteractionRecognitionRuntime:
    def __init__(self,recognizer,catalog,output_dir,interval_s=.5):
        self.recognizer=recognizer;self.catalog=catalog;self.output=Path(output_dir)
        self.interval_s=interval_s;self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='object-recognition')
        self.pending=None;self.generation=0;self.task_id=None;self.store=RecognitionStore(aliases=catalog.label_aliases)
        self.last_submit=-1e9;self.last_frame=None;self.last_debug=None;self.errors=0

    def reset(self,task_id):
        if Path(str(task_id)).name!=str(task_id) or str(task_id) in ['.','..']:
            raise ValueError('Task ID must be a single path component')
        self.generation+=1;self.task_id=str(task_id);self.store=RecognitionStore(aliases=self.catalog.label_aliases)
        self.last_submit=-1e9;self.last_frame=None;self.last_debug=None;self.errors=0
        self.directory=self.output/self.task_id/'recognition';self.directory.mkdir(parents=True,exist_ok=True)

    def wants_frame(self,now,objects):
        return bool(objects) and self.pending is None and now-self.last_submit>=self.interval_s

    def submit(self,camera_data,objects,render_id,now):
        if not self.wants_frame(now,objects):return False
        frame_id=f'{self.task_id}_{render_id}'
        if frame_id==self.last_frame:return False
        if any(camera_data.get(k) is None for k in ['rgba','depth','pointcloud_image']):return False
        captured={k:np.asarray(camera_data[k]).copy() for k in ['rgba','depth','pointcloud_image']}
        self.last_frame=frame_id;self.last_submit=float(now)
        future=self.executor.submit(self.recognizer.recognize,captured,tuple(objects),frame_id,now,self.directory)
        self.pending=(self.generation,future)
        return True

    def poll(self):
        if self.pending is None or not self.pending[1].done():return None
        generation,future=self.pending;self.pending=None
        try:evidence,debug=future.result()
        except Exception as error:
            if generation!=self.generation:return None
            self.errors+=1
            debug={'status':'error','type':type(error).__name__,'message':str(error)}
            with (self.directory/'recognition_events.jsonl').open('a') as stream:stream.write(json.dumps(debug)+'\n')
            return debug
        if generation!=self.generation:return None
        accepted=[e.object_id for e in evidence if self.store.observe(e)]
        debug={**debug,'status':'ok','accepted_evidence_ids':accepted}
        self.errors=0;self.last_debug=debug
        with (self.directory/'recognition_events.jsonl').open('a') as stream:stream.write(json.dumps(debug)+'\n')
        return debug

    def eligible_ids(self,now):return self.catalog.eligible_ids('push_aside',self.store,now)

    def save(self,now):
        data={'required':True,'semantic_source':'rgb_dino_qwen_vl',
              'interactable_categories':list(self.catalog.categories),
              'objects':self.store.snapshot(now),'eligible_ids':sorted(self.eligible_ids(now))}
        (self.directory/'recognized_objects.json').write_text(json.dumps(data,indent=2))
        return data

    def close(self):self.executor.shutdown(wait=True,cancel_futures=True)
