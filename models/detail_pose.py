import os
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
# from ultralytics import YOLO
from init_model import InitModel
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

class DetailedPoseModel(InitModel):
    """세부 자세 탐지 모델"""
    def __init__(self):
        # 모델 초기화
        self.class_mapping = {
            'Thorax_dv_vd':['T_1','T_2','T_3'],
            'Thorax_lateral':['TL_1','TL_2,3,4'],
            'Abdomen_vd':['A_1','A_2'],
            'Abdomen_lateral':['AL_1','AL_2'],
            'Msk_lateral':['ML_1'],
            'Msk_cc':['M_1'],
        }
        self.pose_mapping = {
            0:'Thorax_dv_vd',
            1:'Thorax_lateral',
            2:'Abdomen_vd',
            3:'Abdomen_lateral',
            4:'Msk_cc',
            5:'Msk_lateral',
        }
        
        self.output = [] # 예측 결과
        self.results = [] # 결과 score
        
        self.cls_label = ["A_1","AL_1","T_1","TL_1","C_2"]
        # self.detection_label = ['T_2', 'T_3', 'TL_2,3,4', 'A_2', 'AL_2', 'ML_1', 'M_1', 'C_1']
        self.cls_labels = set([0, 1])
        self.cls_label_number = 2
        self.class_colors = {}
        self.used_text_positions = []
        
    def _get_class_name(self, pose_num):
        pose = self.pose_mapping[pose_num]
        return pose, self.class_mapping[pose]
    
    def generate_random_color(self):
        """무작위 RGB 색상 생성"""
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (r, g, b)

    def get_color_for_class(self, class_name):
        """클래스에 대한 고유한 색상을 가져오거나 생성"""
        if class_name not in self.class_colors:
            # 클래스에 고유한 색상이 없으면 무작위로 생성하고 저장
            self.class_colors[class_name] = self.generate_random_color()
        return self.class_colors[class_name]

    def draw_bbox_on_image(self, image):
        draw = ImageDraw.Draw(image)
        # 폰트 설정 (폰트 크기 50으로 설정)
        try:
            font = ImageFont.truetype("arialbd.ttf", 50)
        except IOError:
            font = ImageFont.load_default()  # 폰트 파일을 찾을 수 없으면 기본 폰트를 사용

        for item in self.output:
            if item['type'] == 'obj':
                confidence = item['confidence']
                class_name = item['class_name']
                x1, y1, x2, y2 = item['bbox'].astype(np.float32)

                # 클래스에 대한 고유한 색상 가져오기
                color = self.get_color_for_class(class_name)

                # 사각형 그리기 (이미지, 좌상단, 우하단, 색)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # 클래스 이름과 confidence 표시
                label = f'{class_name}: {confidence:.2f}'
                # 텍스트 표시 위치
                text_position = (x1, y1 - 60)  # 텍스트가 박스와 겹치지 않도록 조정

                # 텍스트 그리기
                draw.text(text_position, label, fill=color, font=font)

        return image

    def predict(self, pose_num, input_data):
        labels = self._get_class_name(pose_num)[1] # 자세에 해당하는 class name 가져오기
        labels.append("C_1") # 공통 부분 탐지
        labels.append("C_2") # 공통 부분 탐지
        
        for label in labels:
            if label in self.cls_label:
                ### Classification ###
                self.init_cls_model(label, self.cls_labels, self.cls_label_number) # 분류 모델 초기화
                inputs = self.processor(images=input_data, return_tensors="pt")

                # 모델 예측
                with torch.no_grad():  
                    outputs = self.classification_model(**inputs)
                    
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
                self.output.append({"type":"cls",
                                    "class_name":label,
                                    "result":predicted_class})                    
                
            else:
                ### Object Detection ###
                self.init_obj_model(label) # 객체 탐지 모델 초기화
                results = self.detection_model.predict(source=input_data) # image inference

                for result in results:  
                    if result.boxes:  # bounding box가 있는지 확인
                        for box in result.boxes:
                            xyxy = box.xyxy[0].cpu().numpy()  
                            confidence = box.conf[0].cpu().numpy()  
                            class_id = int(box.cls[0].cpu().numpy())  
                            
                            # 클래스 이름
                            class_name = result.names[class_id]  # 예측된 클래스 이름
                            # print(f"객체: {class_name}, 신뢰도: {confidence}, 좌표: {xyxy}")

                            # 원하는 처리 수행 (예: 박스 및 정보 추가)
                            self.output.append({
                                "type":"obj",
                                "class_name": class_name,
                                "confidence": confidence,
                                "bbox": xyxy
                            })
                        
        # 이미지 위에 박스 그리기
        predicted_image=self.draw_bbox_on_image(input_data)
        # predicted_image.show()          
                    
        return self.output, predicted_image
                
                