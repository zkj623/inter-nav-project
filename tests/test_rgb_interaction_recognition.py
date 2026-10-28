import numpy as np
from grutopia_extension.interactive_navigation.rgb_object_recognition import (
    RGBObjectRecognizer,ObjectGeometry,associate_points)
from grutopia_extension.interactive_navigation.interaction_catalog import RecognitionStore,InteractionCatalog


class Backend:
    def __init__(self,label):self.label=label;self.classification_calls=0
    def detect(self,image,caption):return [{'label':'cardboard box','confidence':.8,'bbox':[8,8,56,56]}]
    def segment(self,image,bbox):return np.ones(image.shape[:2],bool)
    def classify(self,image,labels):
        self.classification_calls+=1
        return {'label':self.label,'model':'test-visual-model','raw_text':self.label}


def test_detector_box_word_does_not_override_visual_classifier(tmp_path):
    image=np.zeros((64,64,4),np.uint8);image[:,:,3]=255
    camera={'rgba':image,'depth':np.ones((64,64)),'pointcloud_image':np.zeros((64,64,3))}
    backend=Backend('cabinet');recognizer=RGBObjectRecognizer(backend)
    objects=[ObjectGeometry('B06',(0,0,0),(1,1,1))]
    store=RecognitionStore();catalog=InteractionCatalog()
    for frame in ['f1','f2']:
        values,_=recognizer.recognize(camera,objects,frame,1.,tmp_path)
        for value in values:store.observe(value)
    assert backend.classification_calls==2
    assert store.confirmed('B06',1.).category=='cabinet'
    assert not catalog.eligible_ids('push_aside',store,1.)


def test_depth_association_rejects_foreground_pixels_outside_proposal():
    object_id,_=associate_points(np.ones((50,3))*3,[ObjectGeometry('B06',(0,0,0),(1,1,1))])
    assert object_id is None
