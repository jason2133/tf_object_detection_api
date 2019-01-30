# 모듈들 불러오기
import os
import numpy as np
import tensorflow as tf
import six.moves.urllib as urllib
import tarfile
from PIL import Image
from tqdm import tqdm
from time import gmtime, strftime
import json
import cv2

# Imageio를 사용해 FFmpeg를 플러그인으로 내려받기
try:
    from moviepy.editor import VideoFileClip
except:
    # FFmpeg 다운받기
    import imageio
    imageio.plugins.ffmpeg.download()
    from moviepy.editor import VideoFileClip

# TensorFlow API 프로젝트의 object_detection Directory에 있는 2개의 함수가 필요하기에
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class DetectionObj(object):
    # DetectionObj는 다양한 원천(파일, 웹캠에서 가져온 이미지, 동영상)에서
    # 가져온 이미지에 주석을 달기 위해 Google TensorFlow detection API를 활용하기 적합한 클래스임
    
    def __init__(self, model='ssd_mobilenet_v1_coco_11_06_2017'):
        # 클래스가 인스턴스화될 때 실행될 명령

        # Python Script가 실행될 경로
        self.CURRENT_PATH = os.getcwd()

        # 주석을 저장할 경로 (수정 가능)
        self.TARGET_PATH = self.CURRENT_PATH

        # TensorFlow Model Zoo에서 미리 훈련된 탐지모델 선택
        self.MODELS = ["ssd_mobilenet_v1_coco_11_06_2017",
                        "ssd_inception_v2_coco_11_06_2017",
                        "rfcn_resnet101_coco_11_06_2017",
                        "faster_rcnn_resnet101_coco_11_06_2017",
                        "faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017"]
        
        # 모델이 객체를 탐지할 때 사용할 임계값 설정
        self.THRESHOLD = 0.25 # 실제로 많이 사용하는 임계값은 0.25
        # 주석 처리에 의해 발견될 확률 임계값을 바꿈
        # 완전히 가려지거나 시각적으로 어수선해서 불분명할 객체를 잡아내기에 적절한 임계값은 0.25임

        # 선택한 미리 훈련된 탐지모델이 사용가능한지 확인
        if model in self.MODELS:
            self.MODEL_NAME = model
        else:
            # 사용할 수 없다면 기본 모델로 되돌림
            print("Model not available, reverted to default", self.MODELS[0])
            self.MODEL_NAME = self.MODELS[0]

        # 확정된 텐서플로 모델의 파일명
        self.CKPT_FILE = os.path.join(self.CURRENT_PATH, 'object_detection',
                                      self.MODEL_NAME, 'frozen_inference_graph.pb')

        # 탐지 모델 로딩
        # 디스크에 탐지 모델이 없다면, 인터넷에서 내려받음
        try:
            self.DETECTION_GRAPH = self.load_frozen_model()
        except:
            print("Couldn\'t find", self.MODEL_NAME)
            self.download_frozen_model()
            self.DETECTION_GRAPH = self.load_frozen_model()

        # 탐지 모델에 의해 인식될 클래스 레이블 로딩
        self.NUM_CLASSES = 90
        path_to_labels = os.path.join(self.CURRENT_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
        label_mapping = label_map_util.convert_label_map_to_categories(label_mapping, max_num_classes = self.NUM_CLASSES, use_display_name = True)
        self.LABELS = {item['id'] : item['name'] for item in extracted_categories}
        self.CATEGORY_INDEX = label_map_util.create_category_index(extracted_categories)

        # TensorFlow Session 시작
        self.TF_SESSION = tf.Session(graph = self.DETECTION_GRAPH)

def load_frozen_model(self):
    # Ckpt 파일에 동결된 탐지 모델을 디스크에서 메모리로 적재

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(self.CKPT_FILE, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def download_frozen_model(self):
    # 고정된 탐지 모델이 디스크에 없을 때 인터넷에서 내려받음
    
    def my_hook(t):
        # URLopener를 모니터링하기 위해 tqdm 인스턴스를 감쌈
        last_b = [0]

        def inner(b=1, bsize = 1, tsize = None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        
        return inner

    # 모델을 찾을 수 있는 URL 열기
    model_filename = self.MODEL_NAME + '.tar.gz'
    download_url = 'http://download.tensorflow.org/models/object_detection/'
    opener = urllib.request.URLopener()

    # tqdm 완료 추정을 사용해 모델 내려받기
    print("Downloading ...")
    with tqdm() as t:
        opener.retrieve(download_url + model_filename, model_filename, reporthook = my_hook(t))

    # 내려받은 tar 파일에서 모델 추출하기
    print("Extracting ...")
    tar_file = tarfile.open(model_filename)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.path.join(self.CURRENT_PATH, 'object_detection'))

    




