import os
from transformers import ViTImageProcessor, ViTForImageClassification
from ultralytics import YOLO

class InitModel:
    def __init__(self):
        pass

    def _get_weight_path(self, label):
        ''' 모델 가중치 파일 경로 가져오기 '''
        # return f'./model_weights/{label}'
        return os.path.join("model_weights",label)
    
        # model_weight_dir = os.path.join(os.path.dirname(os.getcwd()), "model_weights")
        # print(model_weight_dir)
        # print('asd',os.path.join(model_weight_dir,label))
        # return os.path.join(model_weight_dir,label)
    
    def init_cls_model(self, label, real_label, label_number):
        ''' 분류 모델 초기화 '''
        self.cls_weight_path = self._get_weight_path(label)
        self.processor =  ViTImageProcessor.from_pretrained(self.cls_weight_path)
        self.classification_model = ViTForImageClassification.from_pretrained(
            self.cls_weight_path,
            num_labels = label_number,
            id2label = {str(i): c for i, c in enumerate(real_label)},
            label2id = {c: str(i) for i, c in enumerate(real_label)}
        )
        
    def init_obj_model(self, label):
        ''' 객체 탐지 모델 초기화 '''
        self.obj_weight_file = os.path.join(self._get_weight_path(label),f'{label}.pt')
        self.detection_model = YOLO(self.obj_weight_file, verbose=False)
        
        
        