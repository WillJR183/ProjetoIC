# Utilities for object detector.
# Utilitários do detector de objetos.
import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict


detection_graph = tf.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
# limite de pontuação para mostrar as caixas delimitadoras.
_score_thresh = 0.27

MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
# Caminho para o gráfico de detecção congelada. Este é o modelo real usado para a detecção de objetos.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
# Lista das sequências de caracteres que são usadas para adicionar o rótulo correto a cada caixa.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
# carrega o mapa de etiqueta.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load a frozen infrerence graph into memory
# Carrega um gráfico de inferência congeladdo na memória.
def load_inference_graph():

    # load frozen tensorflow model into memory
    # carrega o modelo do tensorflow congelado na memória.
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
# Função para desenhar caixas delimitadoras nas imagens.
# Utiliza quatro pontos : left, right, top e bottom.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)

            return [int(left), int(right), int(top), int(bottom)]


# Show fps value on image.
# Exibe a taxa de FPS na imagem.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Actual detection .. generate scores and bounding boxes given an image
# Detecção real, gere pontuações e caixas delimitadoras , dada uma imagem.
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    # Tensores definidos de entrada e saída para detection_graph.
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    # Cada caixa representa uma parte da imagem em que um objeto especifico foi detectado.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    # Cada pontuação representa como o nível de confiaça de cada um dos objetos.
    # A pontução é mostrada na imagem do resultado , junto como o rótulo da classe.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        #Inicialize o fluxo da câmera de video e leia o primeiro quadro do fluxo.
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # Inicialize a variável para ler os quadros do fluxo de video.
        # be stopped
        # Ser parado.
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        # Inicia o thread para ler os quadros do fluxo de video.
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        # Continue fazendo o looping até que o encadeamento seja interrompido.
        while True:
            # if the thread indicator variable is set, stop the thread
            # Se a variável do indicador de linha estiver definida, pare-a.
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            # caso contrário, leia o quadro de texto do fluxo
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        # retorna o quadro mais lido recentemente.
        return self.frame

    def size(self):
        # return size of the capture device
        # retorna o tamanho do dispositivo de captura.
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        # indica que encadeamento deve ser parado.
        self.stopped = True
