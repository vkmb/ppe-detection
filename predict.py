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
    index = None

    for indx in range(len(vest_boxes)):
        index = indx
        vest_flag = is_wearing_hardhat(person_box, vest_boxes[indx], vest_intersection_ratio)
        if vest_flag:
            break

    return index, vest_flag

def check_hardhat(hardhat_boxes, person_box):
    hardhat_flag = False
    hardhat_intersection_ratio = 0.6
    index = None

    for indx in range(len(hardhat_boxes)):
        index = indx
        hardhat_flag = is_wearing_hardhat(person_box, hardhat_boxes[indx], hardhat_intersection_ratio)
        if hardhat_flag:
            break

    return index, hardhat_flag

def is_wearing_hardhat_vest(hardhat_boxes, vest_boxes, person_box):
    hardhat_flag = False
    vest_flag = False
    hardhat_intersection_ratio = 0.6
    vest_intersection_ratio = 0.6
    hardhat_id = None
    vest_id = None
    for hardhatid in range(len(hardhat_boxes)):
        hardhat_id = hardhatid
        hardhat_flag = is_wearing_hardhat(person_box, hardhat_boxes[hardhatid], hardhat_intersection_ratio)
        if hardhat_flag:
            break

    for vestid in range(len(vest_boxes)):
        vest_id = vestid
        vest_flag = is_wearing_vest(person_box, vest_boxes[vestid], vest_intersection_ratio)
        if vest_flag:
            break

    return  hardhat_id,  hardhat_flag, vest_id, vest_flag

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


def predict(graph, labels_list, inference_engine_id=1, streams=None, engine=None):
    
    caps = []
    db_access = False
    min_score_thresh = .4
    category_index = {1: {'id': 1 , 'name': 'Hard Hat'},
                      2: {'id': 2, 'name': 'Saftey Vest'},
                      3: {'id': 3, 'name': 'Person'}}

    try:
        if engine == None:
            engine = generate_db_engine(creds)
        engine.connect()
        db_access = True
    except:
        usr_prompt = input(" * Access to db failed !_! \n    - If you wish to continue type \"y\" : ")
        if usr_prompt != "y":
            exit()
        db_access = False
    
    if streams == None:
        print(" * No streams to capture photons are found !_! \n    - Being pulled into blackhole")
        exit()
    try:
        caps = [int(stream) for stream in streams.values()]
    except:
        pass
    try:
        caps = [cv2.VideoCapture(stream) for stream in streams.values()]
        operating_unit_ids = [stream for stream in streams]
    except:
        if len(caps) == 0:
            print(" * Error accessing streams to capture photons are found !_! \n    - Being pulled into blackhole")
            exit()
        else:
            print(" * Some streams are inaccessible")
        
    # populate the locals
    # model metadata 
    # inference_engine_dict = inference_engine_loader(engine, inference_engine_id)
    # load operating unit metadata
    # operating_unit_dict = operating_unit_loader(engine, operating_unit_id)
    # label_to_predict = label_dict["label_name"]
    for label_list in labels_list:
        st = ""
        if 1 in label_list:
            st += " hard hat "
        if 2 in label_list:
            st += " vest "
        if 3 in label_list:
            st += " illegal entry "
        
        print("MODEL WILL BE DETECTIING", st , "VIOLATIONS")

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
            violation_tracker =  [{"PD" : {"violation": False, "start_time": None, "end_time": None}, "SV" : {"violation": False, "start_time": None, "end_time": None}, "HH" : {"violation": False, "start_time": None, "end_time": None}} for i in caps]
            Flags = [True for i in caps]
            index = 0
            count = 0
            while Flags[index]:
                count += 1
                for cap in caps:
                    current_time = datetime.now()
                    ret, frame = cap.read()
                    index = caps.index(cap)
                    if frame is None or ret == False:
                        print("predict:", "null frame")
                        Flags[index]=False
                        continue
            
                    image_expanded = np.expand_dims(frame, axis=0)
                    output_dict = run_inference_for_single_image(image_expanded, sess, tensor_dict)

                    detection_scores = np.where(output_dict["detection_scores"] > min_score_thresh, True, False)

                    detection_boxes = output_dict["detection_boxes"][detection_scores]
                    detection_classes = output_dict["detection_classes"][detection_scores]
                    
                    vis_utils.visualize_boxes_and_labels_on_image_array(
                        frame,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=4)
                    # cv2.imwrite(f'ppe{count}.jpg', frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                    persons = []
                    # persons2 = []
                
                    if 1 in labels_list[index]:
                        # Area violation
                        print("PD")
                        person_boxes = detection_boxes[np.where(detection_classes == 3)]
                        if not violation_tracker[index]["PD"]["violation"] and len(person_boxes) > 0:
                            violation_tracker[index]["PD"]["violation"] = True
                            violation_tracker[index]["PD"]["start_time"] =  current_time
                            violation_tracker[index]["PD"]["end_time"] = current_time
                            print ("*"*100,"\nViolation Detected : Person Entering Restricted Area\n","*"*100)
                            for box_id in range(len(person_boxes)):
                                #log
                                if db_access:
                                    logging(engine, frame, violation_tracker[index]["PD"]["start_time"], label_id=1, \
                                    inference_engine_id=inference_engine_id, operating_unit_id=operating_unit_ids[index], event_flag=1, index=box_id,\
                                    object_xmin=person_boxes[box_id][0], object_ymin=person_boxes[box_id][1], object_xmax=person_boxes[box_id][2], object_ymax=person_boxes[box_id][3], label_object_pred_threshold=min_score_thresh)
                                
                        elif violation_tracker[index]["PD"]["violation"] and len(person_boxes) == 0 and violation_tracker[index]["PD"]["end_time"] == None:
                            violation_tracker[index]["PD"]["end_time"] = current_time
                            violation_tracker[index]["PD"]["violation"] = False
                        
                        elif len(person_boxes) == 0 and not violation_tracker[index]["PD"]["violation"] and violation_tracker[index]["PD"]["end_time"] != None and current_time - violation_tracker[index]["PD"]["end_time"] > timedelta(seconds=10):
                            violation_tracker[index]["PD"]["start_time"] =  None
                            violation_tracker[index]["PD"]["end_time"] = None
                            if db_access:
                                logging(engine, frame, violation_tracker[index]["PD"]["end_time"], label_id=1, \
                                inference_engine_id=inference_engine_id, operating_unit_id=operating_unit_ids[index], label_object_pred_threshold=min_score_thresh)

                        if 3 in labels_list[index]:
                          
                            vest_boxes = detection_boxes[np.where(detection_classes == 2)]
                            print("SVD")
                            if len(person_boxes) > 0:
                                box_mapper = []
                                for person_box_id in range(len(person_boxes)):
                                    box_id, flag = check_hardhat(vest_boxes, person_boxes[person_box_id])
                                    person_box_index = person_boxes[person_box_id]
                                    box_mapper.append({'box_index': box_id, 'person_box_index':person_box_index})
                                    persons.append(flag)
                            
                                if not violation_tracker[index]["SV"]["violation"] and (not all(persons) or len(vest_boxes)<len(person_boxes)):
                                    violation_tracker[index]["SV"]["violation"] = True
                                    violation_tracker[index]["SV"]["start_time"] =  current_time
                                    violation_tracker[index]["SV"]["end_time"] = current_time
                                    print ("*"*100,"\nViolation Detected : Person Not Wearing Vest\n","*"*100)

                                    for box_id in range(len(vest_boxes)):
                                        if not persons[box_id]:
                                            box = person_boxes[box_id]
                                            if db_access:
                                                logging(engine, frame, violation_tracker[index]["SV"]["start_time"], label_id=3, \
                                                inference_engine_id=inference_engine_id, operating_unit_id=operating_unit_ids[index], event_flag=1, index=box_id,\
                                                object_xmin=box[0], object_ymin=box[1], object_xmax=box[2], object_ymax=box[3], label_object_pred_threshold=min_score_thresh)
                                    
                            elif violation_tracker[index]["SV"]["violation"] and (all(persons) or len(vest_boxes)==len(person_boxes)) and violation_tracker[index]["SV"]["end_time"] == None:
                                violation_tracker[index]["SV"]["end_time"] = current_time
                                violation_tracker[index]["SV"]["violation"] = False
                            
                            elif not violation_tracker[index]["SV"]["violation"] and violation_tracker[index]["SV"]["end_time"] != None and current_time - violation_tracker[index]["SV"]["end_time"] > timedelta(seconds=10):
                                violation_tracker[index]["SV"]["start_time"] =  None
                                violation_tracker[index]["SV"]["end_time"] = None
                                if db_access:
                                    logging(engine, frame, violation_tracker[index]["SV"]["end_time"],  label_id=3, \
                                    inference_engine_id=inference_engine_id, operating_unit_id=operating_unit_ids[index], label_object_pred_threshold=min_score_thresh)
                    

                        if 2 in labels_list[index]:
                            hat_boxes = detection_boxes[np.where(detection_classes == 1)]
                            print("HHD")
                            box_mapper = []
                            
                            for person_box_id in range(len(person_boxes)):
                                    box_id, flag = check_hardhat(hat_boxes, person_boxes[person_box_id])
                                    person_box_index = person_boxes[person_box_id]
                                    box_mapper.append({'box_index': box_id, 'person_box_index':person_box_index})
                                    persons.append(flag)
                            
                            if not violation_tracker[index]["HH"]["violation"] and (not all(persons) or len(hat_boxes)<len(person_boxes)):
                                violation_tracker[index]["HH"]["violation"] = True
                                violation_tracker[index]["HH"]["start_time"] =  current_time
                                violation_tracker[index]["HH"]["end_time"] = current_time
                                print ("*"*100,"\nViolation Detected : Person not wearing hard hat\n","*"*100)

                                for box_id in range(len(box_mapper)):
                                    if not persons[box_id]:
                                        box = person_boxes[box_id]
                                        if db_access:
                                            logging(engine, frame, violation_tracker[index]["HH"]["start_time"], label_id=2, \
                                            inference_engine_id=inference_engine_id, operating_unit_id=operating_unit_ids[index], event_flag=1, index=box_id,\
                                            object_xmin=box[0], object_ymin=box[1], object_xmax=box[2], object_ymax=box[3], label_object_pred_threshold=min_score_thresh)

                            elif violation_tracker[index]["HH"]["violation"] and (all(persons) or len(hat_boxes)==len(person_boxes)) and violation_tracker[index]["HH"]["end_time"] == None:
                                violation_tracker[index]["HH"]["end_time"] = current_time
                                violation_tracker[index]["HH"]["violation"] = False
                            
                            elif len(person_boxes) == 0 and not violation_tracker[index]["HH"]["violation"] and violation_tracker[index]["HH"]["end_time"] != None and current_time - violation_tracker[index]["HH"]["end_time"] > timedelta(seconds=10):
                                violation_tracker[index]["HH"]["start_time"] =  None
                                violation_tracker[index]["HH"]["end_time"] = None
                                if db_access:
                                    logging(engine, frame, violation_tracker[index]["HH"]["end_time"],  label_id=3, \
                                    inference_engine_id=inference_engine_id, operating_unit_id=operating_unit_ids[index], label_object_pred_threshold=min_score_thresh)

                    print(violation_tracker)
                    # elif label_to_predict == "All":
                    #     person_boxes = detection_boxes[np.where(detection_classes == 3)]
                    #     vest_boxes = detection_boxes[np.where(detection_classes == 2)]
                    #     hardhat_boxes = detection_boxes[np.where(detection_classes == 1)]
                    #     box_mapper = []
                    #     for person_box in person_boxes:
                    #             hardhat_index, hardhat_flag, vest_flag, vest_box_index = is_wearing_hardhat_vest(hardhat_boxes, vest_boxes, person_box)
                    #             persons.append(hardhat_flag)
                    #             persons2.append(vest_flag)
                    #             box_mapper.append({'vest_box_index': vest_box_index, 'hard_hat_box_index': hardhat_index,'person_box_index':person_box_index})
                    #     if len(person_boxes) > 0:
                    #         if not violation_tracker[index]["violation"] and (not all(persons) or not all(persons2)):
                    #             violation_tracker[index]["violation"] = True
                    #             violation_tracker[index]["start_time"] =  current_time
                    #             violation_tracker[index]["end_time"] = current_time
                    #             print("violation detected")
                    #             for box_id in range(len(box_mapper)):
                    #                 if not persons[box_id] or not persons2[box_id]:
                    #                     box = person_boxes[box_id]
                    #                     logging(engine, frame, violation_tracker[index]["start_time"], label_id, \
                    #                     inference_engine_id=inference_engine_id, operating_unit_id=operating_unit_ids[index], event_flag=1, index=box_id,\
                    #                     object_xmin=box[0], object_ymin=box[1], object_xmax=box[2], object_ymax=box[3], label_object_pred_threshold=min_score_thresh)
                                    
                                

                    #     elif violation_tracker[index]["violation"] and all(persons) and all(persons2) and violation_tracker[index]["end_time"] == None:
                    #         violation_tracker[index]["end_time"] = current_time
                    #         violation_tracker[index]["violation"] = False
                        
                    #     elif len(person_boxes) == 0 and not violation_tracker[index]["violation"] and violation_tracker[index]["end_time"] != None and current_time - violation_tracker[index]["end_time"] > timedelta(seconds=10):
                    #         violation_tracker[index]["start_time"] =  None
                    #         violation_tracker[index]["end_time"] = None
                    #         print("violation ended")
                    #         logging(engine, frame, violation_tracker[index]["start_time"], label_id, \
                    #                 inference_engine_id=inference_engine_id, operating_unit_id=operating_unit_ids[index], label_object_pred_threshold=min_score_thresh)
                

        print("predict:", "releasing video capture")
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Hardhat and Vest Detection", add_help=True)
    parser.add_argument("--video_file_name", type=str, required=True, help="path to video file, or camera device, i.e /dev/video1 or just 0, 1, ...")
    # parser.add_argument("--model_dir", type=str, required=True, help="path to model directory")
    args = parser.parse_args()

    # frozen_model_path = os.path.join(args.model_dir, "frozen_inference_graph.pb")
    frozen_model_path = "frozen_inference_graph.pb"
    if not os.path.exists(frozen_model_path):
        print("frozen_inference_graph.db file is not exist in model directory")
        exit(-1)
    print("loading model")
    graph = load_model(frozen_model_path)
    
    
    predict(graph, [[1, 2, 3]], streams={0:args.video_file_name})
    
    

if __name__ == '__main__':
    main()

