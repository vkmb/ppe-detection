# -*- coding: utf-8 -*-
"""
This Module is ppe
Example:
    $python video_demo.py
Author: Ming'en Zheng
"""
import os
import cv2
import queue
import config
import base64
import argparse
import numpy as np
import tensorflow as tf
from db_cam_helper import *
import visualization_utils as vis_utils
from datetime import datetime, timedelta
from distutils.version import StrictVersion
from multiprocessing import Process, Queue, Value

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*')


def load_model(inference_model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, sess, tensor_dict):
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict


def is_wearing_hardhat(person_box, hardhat_box, intersection_ratio):
    xA = max(person_box[0], hardhat_box[0])
    yA = max(person_box[1], hardhat_box[1])
    xB = min(person_box[2], hardhat_box[2])
    yB = min(person_box[3], hardhat_box[3])

    interArea = max(0, xB - xA ) * max(0, yB - yA )

    hardhat_size = (hardhat_box[2] - hardhat_box[0]) * (hardhat_box[3] - hardhat_box[1])

    if interArea / hardhat_size > intersection_ratio:
        return True
    else:
        return False


def is_wearing_vest(person_box, vest_box, vest_intersection_ratio):
    xA = max(person_box[0], vest_box[0])
    yA = max(person_box[1], vest_box[1])
    xB = min(person_box[2], vest_box[2])
    yB = min(person_box[3], vest_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    vest_size = (vest_box[2] - vest_box[0]) * (vest_box[3] - vest_box[1])

    if interArea / vest_size > vest_intersection_ratio:
        return True
    else:
        return False


def check_vest(vest_boxes, person_box):
    vest_flag = False
    vest_intersection_ratio = 0.6

    for vest_box in vest_boxes:
        vest_flag = is_wearing_hardhat(person_box, vest_box, vest_intersection_ratio)
        if vest_flag:
            break

    return vest_boxes.index(vest_box), vest_flag

def check_hardhat(hardhat_boxes, person_box):
    hardhat_flag = False
    hardhat_intersection_ratio = 0.6

    for hardhat_box in hardhat_boxes:
        hardhat_flag = is_wearing_hardhat(person_box, hardhat_box, hardhat_intersection_ratio)
        if hardhat_flag:
            break

    return hardhat_boxes.index(hardhat_box), hardhat_flag

def is_wearing_hardhat_vest(hardhat_boxes, vest_boxes, person_box):
    hardhat_flag = False
    vest_flag = False
    hardhat_intersection_ratio = 0.6
    vest_intersection_ratio = 0.6

    for hardhat_box in hardhat_boxes:
        hardhat_flag = is_wearing_hardhat(person_box, hardhat_box, hardhat_intersection_ratio)
        if hardhat_flag:
            break

    for vest_box in vest_boxes:
        vest_flag = is_wearing_vest(person_box, vest_box, vest_intersection_ratio)
        if vest_flag:
            break

    return  hardhat_boxes.index(hardhat_box),  hardhat_flag, vest_boxes.index(vest_box), vest_flag


# def post_message(camera_id, output_dict, image, min_score_thresh):
#     message = dict()
#     message["timestamp"] = int(time.time() * 1000)
#     message["cameraId"] = camera_id

#     image_info = {}
#     image_info["height"] = image.shape[0]
#     image_info["width"] = image.shape[1]
#     image_info["format"] = "jpeg"

#     success, encoded_image = cv2.imencode('.jpg', image)
#     content = encoded_image.tobytes()
#     image_info["raw"] = base64.b64encode(content).decode('utf-8')

#     message["image"] = image_info

#     detection_scores = np.where(output_dict["detection_scores"] > min_score_thresh, True, False)

#     detection_boxes = output_dict["detection_boxes"][detection_scores]
#     detection_classes = output_dict["detection_classes"][detection_scores]

#     hardhat_boxes = detection_boxes[np.where(detection_classes == 1)]
#     vest_boxes = detection_boxes[np.where(detection_classes == 2)]
#     person_boxes = detection_boxes[np.where(detection_classes == 3)]

#     persons = []
#     for person_box in person_boxes:
#         person = dict()
#         person["hardhat"], person["vest"] = is_wearing_hardhat_vest(hardhat_boxes, vest_boxes, person_box)
#         persons.append(person)

#     message["persons"] = persons
   
#     if len(persons) == 0:
#         return False

#     print(message["persons"])
#     try:
#         headers = {'Content-type': 'application/json'}
#         if len(persons):
#             result = requests.post(config.detection_api, json=message, headers=headers)
#             print(result)
#             return True
#     except requests.exceptions.ConnectionError:
#         print("Connect to backend failed")
#     return False



def logging(image, timestamp, label_id, inference_engine_id, operating_unit_id, \
        event_flag=0, index="", current_flag=1, active_flag=1, delete_flag=0, object_xmin=0, object_ymin=0, \
        object_xmax=0, object_ymax=0, label_object_pred_threshold=0, label_object_pred_confidence=0 ):
    
    frame_dict, object_dtl_dict = {}, {}
    time = timestamp.strftime("%d/%M/%Y %I:%M:%S %p")
    file_name = f'{operating_unit_id} {inference_engine_id} {label_id} {time} {index}'
    cv2.imwrite(file_name+".jpg", image)
    frame_dict['frame_name'] = file_name
    frame_dict['frame_stored_location'] = os.path.abspath(file_name+".jpg")
    frame_dict['frame_stored_encoding'] = "JPG"
    frame_dict['frame_local_timestamp'] = f"\'{time}\'"
    frame_dict['frame_local_time_zone'] = 'IST'
    frame_dict['frame_size'] = ", ".join(image.shape)
    frame_dict['created_by'] = inference_engine_id
    frame_dict['created_date'] =  f"\'{time}\'"
    frame_dict['active_flag'] = active_flag
    frame_dict['current_flag'] = current_flag
    frame_dict['delete_flag'] = delete_flag
    frame_id = frame_writer(engine, frame_dict)
    
    if frame_id == None:
        return None
    
    if event_flag:
        object_dtl_dict['frame_id'] = frame_id
        object_dtl_dict['object_loc_id'] = operating_unit_id
        object_dtl_dict['label_id'] = label_id
        object_dtl_dict['created_by'] = inference_engine_id
        object_dtl_dict['created_date'] =  f"\'{time}\'"
        object_dtl_dict['active_flag'] = active_flag
        object_dtl_dict['current_flag'] = current_flag
        object_dtl_dict['delete_flag'] = delete_flag
        object_dtl_dict['object_xmin'] = object_xmin
        object_dtl_dict['object_ymin'] = object_ymin
        object_dtl_dict['object_xmax'] = object_xmax
        object_dtl_dict['object_ymax'] = object_ymax
        object_dtl_dict['label_object_pred_threshold'] = label_object_pred_threshold
        object_dtl_dict['label_object_pred_confidence'] = label_object_pred_confidence
        object_dtl_id = object_dtl_writer(engine, object_dtl_dict)

    if object_dtl_id == None:
        return None
    
    data_dict = {}
    data_dict["video_id"] = -1
    data_dict["video_dtl_seq"] = -1
    data_dict["inference_engine_id"] = inference_engine_id
    data_dict["operating_unit_id"] = operating_unit_id
    data_dict["operating_unit_seq"] = operating_unit_id
    data_dict["frame_id"] = frame_id
    data_dict["label_id"] = label_id
    data_dict["label_seq"] = label_id
    data_dict["event_processed_time_zone"] = "IST"
    data_dict["event_processed_local_time"] = f"\'{time}\'"
    data_dict["event_flag"] = event_flag
    data_dict["created_date"] = f"\'{time}\'"
    data_dict["created_by"] = inference_engine_id
    data_dict["current_flag"] = current_flag
    data_dict["active_flag"] = active_flag
    data_dict["delete_flag"] = delete_flag
    
    event_log_dtl_writer(engine, data_dict)


def image_processing(graph, category_index, image_file_name, show_video_window):

    img = cv2.imread(image_file_name)
    image_expanded = np.expand_dims(img, axis=0)

    with graph.as_default():
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        with tf.Session() as sess:
            output_dict = run_inference_for_single_image(image_expanded, sess, tensor_dict)

            vis_utils.visualize_boxes_and_labels_on_image_array(
                img,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=4)

            if show_video_window:
                cv2.imshow('ppe', img)
                cv2.waitKey(5000)


def predict(graph, category_index, video_file_name):
    
    cap = cv2.VideoCapture(video_file_name)
    min_score_thresh = .5
    engine = generate_db_engine(creds)
    # get confifuration
    seq, operating_unit_id, inference_engine_id, label_id = ou_inference_loader(engine)
    # load label meta data
    label_dict = label_loader(engine, label_id)
    # model metadata 
    # inference_engine_dict = inference_engine_loader(engine, inference_engine_id)
    # load operating unit metadata
    # operating_unit_dict = operating_unit_loader(engine, operating_unit_id)
    label_to_predict = label_dict["label_name"]
    with graph.as_default():
        print("predict:", "default tensorflow graph")
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        with tf.Session() as sess:
            print("predict:", "tensorflow session")
            violation_tracker =  {"violation": False, "start_time": None, "end_time": None}
            while True:
                current_time = datetime.now()
                ret, frame = cap.read()

                if frame is None or ret == False:
                    print("predict:", "null frame")
                    break
            
                image_expanded = np.expand_dims(frame, axis=0)
                output_dict = run_inference_for_single_image(image_expanded, sess, tensor_dict)

                detection_scores = np.where(output_dict["detection_scores"] > min_score_thresh, True, False)

                detection_boxes = output_dict["detection_boxes"][detection_scores]
                detection_classes = output_dict["detection_classes"][detection_scores]

                persons = []
                persons2 = []
                
                if label_to_predict == "Human":
                    # Area violation
                    
                    boxes = detection_boxes[np.where(detection_classes == 3)]
                    if not violation_tracker["violation"] and len(boxes) > 0:
                        violation_tracker["violation"] = True
                        violation_tracker["start_time"] =  current_time
                        violation_tracker["end_time"] = current_time
                        for box in boxes:
                            #log
                            logging(frame, str(violation_tracker["start_time"]), label_id, \
                            inference_engine_id, operating_unit_id, event_flag=1, index=boxes.index(box),\
                            object_xmin=box[0], object_ymin=box[1], object_xmax=box[2], object_ymax=box[3], label_object_pred_threshold=min_score_thresh)
                            
                    elif violation_tracker["violation"] and len(boxes) == 0 and violation_tracker["end_time"] == None:
                        violation_tracker["end_time"] = current_time
                        violation_tracker["violation"] = False
                    
                    elif len(boxes) == 0 and not violation_tracker["violation"] and current_time - violation_tracker["end_time"] > timedelta(seconds=10):
                        violation_tracker["start_time"] =  None
                        violation_tracker["end_time"] = None
                        logging(frame, str(violation_tracker["start_time"]), label_id, \
                                inference_engine_id, operating_unit_id, label_object_pred_threshold=min_score_thresh)

                elif label_to_predict == "Saftey Vest":
                    person_boxes = detection_boxes[np.where(detection_classes == 3)]
                    vest_boxes = detection_boxes[np.where(detection_classes == 2)]

                    if len(person_boxes) > 0:
                        box_mapper = []
                        for person_box in person_boxes:
                            box_id, flag = check_vest(vest_boxes, person_box)
                            person_box_index = person_boxes.index(person_box)
                            box_mapper.append({'box_index': box_id, 'person_box_index':person_box_index})
                            persons.append(flag)
                    
                        if not violation_tracker["violation"] and not all(persons):
                            violation_tracker["violation"] = True
                            violation_tracker["start_time"] =  current_time
                            violation_tracker["end_time"] = current_time
                            
                            for _box in box_mapper:
                                box = person_boxes[_box['person_box_index']]
                                logging(frame, str(violation_tracker["start_time"]), label_id, \
                                inference_engine_id, operating_unit_id, event_flag=1, index=boxes.index(box),\
                                object_xmin=box[0], object_ymin=box[1], object_xmax=box[2], object_ymax=box[3], label_object_pred_threshold=min_score_thresh)
                            
                    elif violation_tracker["violation"] and all(persons) and violation_tracker["end_time"] == None:
                        violation_tracker["end_time"] = current_time
                        violation_tracker["violation"] = False
                    
                    elif not violation_tracker["violation"] and current_time - violation_tracker["end_time"] > timedelta(seconds=10):
                        violation_tracker["start_time"] =  None
                        violation_tracker["end_time"] = None
                        logging(frame, str(violation_tracker["start_time"]), label_id, \
                                inference_engine_id, operating_unit_id, label_object_pred_threshold=min_score_thresh)
                

                elif label_to_predict == "Hard Hat":
                    person_boxes = detection_boxes[np.where(detection_classes == 3)]
                    hat_boxes = detection_boxes[np.where(detection_classes == 1)]
                    
                    box_mapper = []
                    for person_box in person_boxes:
                            box_id, flag = persons.append(check_hardhat(hat_boxes, person_box))
                            person_box_index = person_boxes.index(person_box)
                            box_mapper.append({'box_index': box_id, 'person_box_index':person_box_index})
                            persons.append(flag)
                            
                    
                    if len(person_boxes) > 0:
                        if not violation_tracker["violation"] and not all(persons):
                            violation_tracker["violation"] = True
                            violation_tracker["start_time"] =  current_time
                            violation_tracker["end_time"] = current_time

                            for _box in box_mapper:
                                box = person_boxes[_box['person_box_index']]
                                logging(frame, str(violation_tracker["start_time"]), label_id, \
                                inference_engine_id, operating_unit_id, event_flag=1, index=boxes.index(box),\
                                object_xmin=box[0], object_ymin=box[1], object_xmax=box[2], object_ymax=box[3], label_object_pred_threshold=min_score_thresh)

                    elif violation_tracker["violation"] and all(persons) and violation_tracker["end_time"] == None:
                        violation_tracker["end_time"] = current_time
                        violation_tracker["violation"] = False
                    
                    elif len(boxes) == 0 and not violation_tracker["violation"] and current_time - violation_tracker["end_time"] > timedelta(seconds=intimate_after):
                        violation_tracker["start_time"] =  None
                        violation_tracker["end_time"] = None
                        logging(frame, str(violation_tracker["start_time"]), label_id, \
                                inference_engine_id, operating_unit_id, label_object_pred_threshold=min_score_thresh)

                elif label_to_predict == "All":
                    person_boxes = detection_boxes[np.where(detection_classes == 3)]
                    vest_boxes = detection_boxes[np.where(detection_classes == 2)]
                    hardhat_boxes = detection_boxes[np.where(detection_classes == 1)]
                    box_mapper = []
                    for person_box in person_boxes:
                            hardhat_index, hardhat_flag, vest_flag, vest_box_index = is_wearing_hardhat_vest(hardhat_boxes, vest_boxes, person_box)
                            persons.append(hardhat_flag)
                            persons2.append(vest_flag)
                            box_mapper.append({'vest_box_index': vest_box_index, 'hard_hat_box_index': hardhat_index,'person_box_index':person_box_index})
                    if len(person_boxes) > 0:
                        if not violation_tracker["violation"] and (not all(persons) or not all(persons2)):
                            violation_tracker["violation"] = True
                            violation_tracker["start_time"] =  current_time
                            violation_tracker["end_time"] = current_time
                            for _box in box_mapper:
                                box = person_boxes[_box['person_box_index']]
                                logging(frame, str(violation_tracker["start_time"]), label_id, \
                                inference_engine_id, operating_unit_id, event_flag=1, index=boxes.index(box),\
                                object_xmin=box[0], object_ymin=box[1], object_xmax=box[2], object_ymax=box[3], label_object_pred_threshold=min_score_thresh)

                    elif violation_tracker["violation"] and (not all(persons) or not all(persons2)) and violation_tracker["end_time"] == None:
                        violation_tracker["end_time"] = current_time
                        violation_tracker["violation"] = False
                    
                    elif len(boxes) == 0 and not violation_tracker["violation"] and current_time - violation_tracker["end_time"] > timedelta(seconds=intimate_after):
                        violation_tracker["start_time"] =  None
                        violation_tracker["end_time"] = None
                        logging(frame, str(violation_tracker["start_time"]), label_id, \
                                inference_engine_id, operating_unit_id, label_object_pred_threshold=min_score_thresh)
                        

    print("predict:", "releasing video capture")
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Hardhat and Vest Detection", add_help=True)
    parser.add_argument("--model_dir", type=str, required=True, const="", help="path to model directory")
    parser.add_argument("--video_file_name", type=str, required=True, help="path to video file, or camera device, i.e /dev/video1")
    args = parser.parse_args()

    frozen_model_path = os.path.join(args.model_dir, "frozen_inference_graph.pb")
    if not os.path.exists(frozen_model_path):
        print("frozen_inference_graph.db file is not exist in model directory")
        exit(-1)
    print("loading model")
    graph = load_model(frozen_model_path)
    category_index = {1: {'id': 1 , 'name': 'Hard Hat'},
                      2: {'id': 2, 'name': 'Saftey Vest'},
                      3: {'id': 3, 'name': 'Human'}}
    
    predict(graph, category_index, args.video_file_name)
    
    

if __name__ == '__main__':
    main()

