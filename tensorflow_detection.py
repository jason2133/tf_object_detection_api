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
    
    # load_image_from_disk와 load_image_into_numpy_arrray는 
    # 디스크에서 이미지를 가져와 이 프로젝트에 있는 텐서플로 모델이 처리하기에 적합한 Numpy 배열로 변환하기 위해 필요함

    def load_image_from_disk(self, image_path):
        return Image.open(image_path)

    def load_image_into_numpy_array(self, image):
        try:
            (im_width, im_height) = image.size # 이미지 크기
            return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.unit8)
        except:
            # 이전 프로시저가 실패하면 우리는 이미지가 이미 Numpy ndarray라고 생각한다
            return image
    
    # detect 함수 : 범주 분류 기능의 핵심
    # 이 함수는 처리할 이미지 리스트만 받음
    # Boo Plug인 annotate_on_image는 제공된 이미지에 윤곽 상자와 주석을 직접 시각화하도록 Script에 지시함

    def detect(self, images, annotate_on_image = True):
        # 이미지 리스트를 처리해서 탐지 모델에 제공하고
        # 모델로부터 이미지에 표시될 점수, 윤곽 상자, 예측 범주를 가져옴

        if type(images) is not list:
            images = [images]
        results = list() # 리스트 처리

        for image in images:
            # 이미지를 배열 기반으로 나타내면
            # 상자와 상자 레이블을 가지고 결과 이미지를 준비하기 위해 나중에 사용될 것임
            image_np = self.load_image_into_numpy_array(image)

            # 모델은 [1, None, None, 3] 형상을 갖는 이미지를 기대하므로 차원을 확장함
            image_np_expanded = np.expand_dims(image_np, axis = 0)
            image_tensor = self.DETECTION_GRAPH.get_tensor_by_name('image_tensor:0')

            # 각 상자는 이미지에서 특정 사물이 탐지된 부분을 나타냄
            boxes = self.DETECTION_GRAPH.get_tensor_by_name('detection_boxes:0')

            # 점수는 각 객체에 대한 신뢰 수준을 나타냄
            # 점수는 범주 레이블과 함께 결과 이미지에 나타낼 수 있음
            scores = self.DETECTION_GRAPH.get_tensor_by_name('detection_scores:0')
            classes = self.DETECTION_GRAPH.get_tensor_by_name('detection_classes:0')
            num_detections = self.DETECTION_GRAPH.get_tensor_by_name('num_detections:0')

            # 여기서 실제로 객체가 탐지됨
            (boxes, scores, classes, num_detections) = self.TF_SESSION.run(
                [boxes, scores, classes, num_detections],
                feed_dict = {image_tensor : image_np_expanded}
            )

            if annotate_on_image:
                new_image = self.detection_on_image(image_np, boxes, scores, classes)
                results.append((new_image, boxes, scores, classes, num_detections))
            else:
                results.append((image_np, boxes, scores, classes, num_detections))
        return results


    def detection_on_image(self, image_np, boxes, scores, classes):
        # 이미지에 탐지된 범주로 탐지 상자 두기
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.CATEGORY_INDEX,
            use_normalized_coordinates=True,
            line_thickness = 8
        )

        return image_np

    def visualize_image(self, image_np, image_size = (400, 300), latency = 3, bluish_correction=True):
        # blusih_correction : RGB를 BGR 형식으로 바꿈
        # BGR 형식은 OpenCV 라이브러리의 표준 형식임
        height, width, depth = image_np.shape
        reshaper = height / float(image_size[0])
        width = int(width / reshaper)
        height = int(height / reshaper)
        id_img = 'preview_' + str(np.sum(image_np))
        cv2.startWindowThread()
        cv2.namedWindow(id_img, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(id_img, width, height)
        if bluish_correction:
            RGB_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            cv2.imshow(id_img, RGB_img)
        else:
            cv2.imshow(id_img, image_np)
        cv2.waitKey(latency * 1000)

    # 주석은 serialize_annotations 함수에 의해 준비되고 디스크에 기록됨
    def serialize_annotations(self, boxes, scores, classes, filename = 'data.json'):
        # 주석을 디스크 및 JSON 파일에 저장

        threshold = self.THRESHOLD
        valid = [position for position, score in enumerate(scores[0]) if score > threshold]
        if len(valid) > 0:
            valid_scores = scores[0][valid].tolist()
            valid_boxes = boxes[0][valid].tolist()
            valid_class = [self.LABELS[int(a_class)] for a_class in classes[0][valid]]
            with open(filename, 'w') as outfile:
                json_data = json.dumps({'classes': valid_class, 'boxes':valid_boxes, 'scores': valid_scores})
                json.dump(json_data, outfile)
    
    # get_time은 파일명에 사용되기 편리하도록 실제 시간을 문자열로 변환함
    def get_time(self):
        # 실제 날짜와 시간을 보고하는 문자열 반환
        return strftime("%Y-%m-%d_%Hh%Mm%Ss", gmtime())
    
    def annotate_photogram(self, photogram):
        # 동영상에서 가져온 사진을 탐지 범주에 해당하는 윤곽 상자로 주석 달기
        new_photogram, boxes, scores, classes, num_detections = self.detect(photogram)[0]
        return new_photogram

    # capture_webcam 함수는 cv2.VideoCapture 기능을 사용해 웹캠에서 이미지를 얻음
    # 웹캠은 사진이 찍히는 환경의 밝기 조건에 먼저 적응해야하기 때문에 객체 탐지 프로시저에 사용될 사진을 얻기 전에 처음 찍은 사진 몇장은 삭제함
    # 이 방식으로 웹캠은 언제나 밝기 설정을 조정할 수 있음
    def capture_webcam(self):
        # 통합된 웹캠에서 이미지 캡처하기

        def get_image(device):
            # 카메라에서 단일 이미지를 캡처해서 PIL 형식으로 반환하는 내부 함수
            retval, im = device.read()
            return im

        # 통합된 웹캠 설정하기
        camera_port = 0

        # 카메라가 주변 빛에 맞춰 조정하기 때문에 버려야 할 프레임 개수
        ramp_frames = 30

        # cv2.VideoCapture로 웹캠 초기화
        camera = cv2.VideoCapture(camera_port)

        # 카메라 램프 조절 - 카메라를 적절한 밝기 수준에 맞추기 때문에 이 프레임들은 모두 제거
        print("Setting the WebCam")
        for i in range(ramp_frames):
            _ = get_image(camera)
        
        # 스냅샷 가져오기
        print("Now taking a snapshot ...", end = '')
        camera_capture = get_image(camera)
        print('Done')

        # 카메라를 해제하고 재활용할 수 있게 만듦
        del (camera)
        return camera_capture

    # File_pipeline은 Stroage에서 이미지를 로딩하고 이를 시각화하고 주석을 달기 위해 필요한 모든 단계로 구성됨
    def file_pipeline(self, images, visualize = True):
        # 디스크로부터 로딩할 이미지 리스트를 처리하고 주석을 달기 위한 파이프라인
        if type(images) is not list:
            images = [images]
        for filename in images:
            single_image = self.load_image_from_disk(filename)
            for new_image, boxes, scores, classes, num_detections in self.detect(single_image):
                self.serialize_annotations(boxes, scores, classes, filename = filename + ".json")
                if visualize:
                    self.visualize_image(new_image)

    # Video_pipeline은 단순히 동영상을 윤곽 상자로 주석 달기 위해 필요한 모든 단계를 배치하고 작업이 완료되면 결과를 디스크에 저장함
    def Video_pipeline(self, video, audio=False):
        # 디스크 상의 동영상을 처리해서 윤곽 상자로 주석을 달기 위한 파이프라인
        # 결과는 주석이 추가된 새로운 동영상임
    
        clip = VideoFileClip(video)
        new_video = video.split('/')
        new_video[-1] = "annotated_" + new_video[-1]
        new_video = '/'.join(new_video)
        print("Saving annotated video to", new_video)
        video_annotation = clip.fl_image(self.annotate_photogram)
        video_annotation.write_videofile(new_video, audio = audio)

    # Webcam_pipeline은 웹캠에서 얻은 이미지에 주석을 달고자 할 때 필요한 모든 단계를 배치한 함수
    def Webcam_pipeline(self):
        # 내부 웹캠에서 얻은 이미지를 처리해서 주석을 달고 JSON 파일을 디스크에 저장하는 파이프라인
        webcam_image = self.capture_webcam()
        filename = "webcam_" + self.get_time()
        saving_path = os.path.join(self.CURRENT_PATH, filename + ".jpg")
        cv2.imwrite(saving_path, webcam_image)
        new_image, boxes, scores, classes, num_detections = self.detect(webcam_image)[0]
        json_obj = {'classes' : classes, 'boxes' : boxes, 'scores' : scores}
        self.serialize_annotations(boxes, scores, classes, filename = filename + ".json")
        self.visualize_image(new_image, blusih_correction=False)







    




