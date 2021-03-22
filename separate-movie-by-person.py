import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/Collectors.mp4', 'path to input video')
flags.DEFINE_float('iou', 0.7, 'iou threshold')
flags.DEFINE_float('score', 0.90, 'score threshold')


def main(_argv):
    global fourcc
    global cap_fps
    global width
    global height
    global out
    global start
    print("start")
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    point = 0
    frame_count = 0
    start = 0
    print("Video from: ", video_path )
    cap = cv2.VideoCapture(video_path)
    out = 0
    width = int(cap.get(3))
    height = int(cap.get(4))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    global frame_threshold
    frame_threshold = 60

    while True:
        return_value, frame = cap.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            break
            
        pos_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        
        print(pos_frames)
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        print(pos_frames)
        #print(valid_detections.numpy())
        point, frame_count, out, start = check_frame(frame, pos_frames, valid_detections.numpy()[0], point, frame_count, out, start)
        '''
        image = utils.draw_bbox(frame, pred_bbox)

        result = np.asarray(image)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        '''
        #out.write(result)
        
    cap.release()


def person_check(frame, pos_frames, person_count, out, point, start, frame_count):
    if person_count > 3:
        person_count = 4

    if point == 0:
        out, start = make_new_video(frame, pos_frames, person_count, out)
        frame_count += 1
        point += 1
    elif point < frame_threshold:
        if start != person_count:
            point += 1
        out.write(frame)
        frame_count += 1
    else:
        point = 0
        frame_count = 0
        out.release()

    return frame, pos_frames, person_count, out, point, start, frame_count

def check_frame(frame, pos_frames, person_count, point, frame_count, out, start):

    frame = np.asarray(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if person_count < 2:
        if frame_count > 0:
            out.write(frame)
            point += 1
    else:
        frame, pos_frames, person_count, out, point, start, frame_count = person_check(frame, pos_frames, person_count, out, point, start, frame_count)

    return point, frame_count, out, start

def make_new_video(frame, pos_frames, person_count, out):
    out = cv2.VideoWriter("./result_movie/"+str(person_count)+"/"+str(pos_frames)+".avi", fourcc, int(cap_fps), (width,height))
    start = person_count
    return out, start


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass