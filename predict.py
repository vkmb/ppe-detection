import os
import cv2
import queue
import config
import base64
import argparse
import numpy as np
import tensorflow as tf
from db_cam_helper import *
from threading import Thread
import visualization_utils as vis_utils
from datetime import datetime, timedelta
from distutils.version import StrictVersion

violation_trackers = None
db_access = False


if StrictVersion(tf.__version__) < StrictVersion("1.12.0"):
    raise ImportError("Please upgrade your TensorFlow installation to v1.12.*")


def load_model(inference_model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_model_path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")
    return detection_graph


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, sess, tensor_dict):
    image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

    output_dict["num_detections"] = int(output_dict["num_detections"][0])
    output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(
        np.int64
    )
    output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
    output_dict["detection_scores"] = output_dict["detection_scores"][0]

    return output_dict


def is_wearing_hardhat(person_box, hardhat_box, intersection_ratio):
    xA = max(person_box[0], hardhat_box[0])
    yA = max(person_box[1], hardhat_box[1])
    xB = min(person_box[2], hardhat_box[2])
    yB = min(person_box[3], hardhat_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

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
        vest_flag = is_wearing_hardhat(
            person_box, vest_boxes[indx], vest_intersection_ratio
        )
        if vest_flag:
            break

    return index, vest_flag


def check_hardhat(hardhat_boxes, person_box):
    hardhat_flag = False
    hardhat_intersection_ratio = 0.6
    index = None

    for indx in range(len(hardhat_boxes)):
        index = indx
        hardhat_flag = is_wearing_hardhat(
            person_box, hardhat_boxes[indx], hardhat_intersection_ratio
        )
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
        hardhat_flag = is_wearing_hardhat(
            person_box, hardhat_boxes[hardhatid], hardhat_intersection_ratio
        )
        if hardhat_flag:
            break

    for vestid in range(len(vest_boxes)):
        vest_id = vestid
        vest_flag = is_wearing_vest(
            person_box, vest_boxes[vestid], vest_intersection_ratio
        )
        if vest_flag:
            break

    return hardhat_id, hardhat_flag, vest_id, vest_flag


def image_processing(graph, category_index, image_file_name, show_video_window):

    img = cv2.imread(image_file_name)
    image_expanded = np.expand_dims(img, axis=0)

    with graph.as_default():
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
            "detection_masks",
        ]:
            tensor_name = key + ":0"
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name
                )
        with tf.Session() as sess:
            output_dict = run_inference_for_single_image(
                image_expanded, sess, tensor_dict
            )

            vis_utils.visualize_boxes_and_labels_on_image_array(
                img,
                output_dict["detection_boxes"],
                output_dict["detection_classes"],
                output_dict["detection_scores"],
                category_index,
                instance_masks=output_dict.get("detection_masks"),
                use_normalized_coordinates=True,
                line_thickness=4,
            )

            if show_video_window:
                cv2.imshow("ppe", img)
                cv2.waitKey(5000)


def predict(
    graph,
    labels_list,
    inference_engine_id=1,
    interval_time=1,
    streams=None,
    engine=None,
    skip_frame_count=20
):
    global violation_trackers, db_access
    caps = []
    db_access = False
    min_score_thresh = 0.5
    category_index = {
        1: {"id": 1, "name": "Hard Hat"},
        2: {"id": 2, "name": "Saftey Vest"},
        3: {"id": 3, "name": "Person"},
    }

    try:
        if engine == None:
            engine = generate_db_engine(creds)
        engine.connect()
        db_access = True
    except:
        usr_prompt = input(
            ' * Access to db failed !_! \n    - If you wish to continue type "y" : '
        )
        if usr_prompt != "y":
            exit()
        db_access = False

    if streams == None:
        print(
            " * No streams to capture photons are found !_! \n    - Being pulled into blackhole"
        )
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
            print(
                " * Error accessing streams to capture photons are found !_! \n    - Being pulled into blackhole"
            )
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

        print("MODEL WILL BE DETECTIING", st, "VIOLATIONS")

    with graph.as_default():
        print("predict:", "default tensorflow graph")
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
            "detection_masks",
        ]:
            tensor_name = key + ":0"
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name
                )
        with tf.Session() as sess:
            print("predict:", "tensorflow session")
            violation_trackers = [
                {
                    "PD": {"violation": False, "start_time": None, "end_time": None},
                    "SV": {"violation": False, "start_time": None, "end_time": None},
                    "HH": {"violation": False, "start_time": None, "end_time": None},
                }
                for i in caps
            ]
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
                        Flags[index] = False
                        continue

                    if skip_frame_count == 0:
                        pass
                    elif count % skip_frame_count == 0:
                        pass
                    else:
                        continue

                    image_expanded = np.expand_dims(frame, axis=0)
                    output_dict = run_inference_for_single_image(
                        image_expanded, sess, tensor_dict
                    )

                    detection_scores = np.where(
                        output_dict["detection_scores"] > min_score_thresh, True, False
                    )

                    detection_boxes = output_dict["detection_boxes"][detection_scores]
                    detection_classes = output_dict["detection_classes"][
                        detection_scores
                    ]
                    hat_boxes = detection_boxes[np.where(detection_classes == 1)]
                    vest_boxes = detection_boxes[np.where(detection_classes == 2)]
                    person_boxes = detection_boxes[np.where(detection_classes == 3)]
                    frame = vis_utils.visualize_boxes_and_labels_on_image_array(
                        frame,
                        output_dict["detection_boxes"],
                        output_dict["detection_classes"],
                        output_dict["detection_scores"],
                        category_index,
                        instance_masks=output_dict.get("detection_masks"),
                        use_normalized_coordinates=True,
                        line_thickness=2,
                    )
                    persons_wearing_hats = []
                    persons_wearing_vests = []

                    if 1 in labels_list[index]:
                        print(
                            "PD",
                            violation_trackers[index]["PD"]["end_time"],
                            violation_trackers[index]["PD"]["start_time"],
                            current_time,
                        )
                        label_id = 1
                        if (not violation_trackers[index]["PD"]["violation"]) and len(
                            person_boxes
                        ) != 0:
                            violation_trackers[index]["PD"]["violation"] = True
                            violation_trackers[index]["PD"]["start_time"] = current_time
                            violation_trackers[index]["PD"]["end_time"] = None

                            print(
                                "*" * 100,
                                "\nViolation Detected : Person entering unathourised zone\n",
                                "*" * 100,
                            )

                            for box_id in range(len(person_boxes)):
                                # cv2.imwrite(f"ppe{count}_pd_error.jpg", frame)
                                event_flag = 1
                                box = person_boxes[box_id]
                                if db_access:
                                    Thread(
                                        target=logging,
                                        args=(
                                            engine,
                                            frame,
                                            violation_trackers[index]["PD"][
                                                "start_time"
                                            ],
                                        ),
                                        kwargs={
                                            "label_id": label_id,
                                            "inference_engine_id": inference_engine_id,
                                            "label_object_pred_threshold": min_score_thresh,
                                            "operating_unit_id": operating_unit_ids[
                                                index
                                            ],
                                            "event_flag": event_flag,
                                            "index": box_id,
                                            "object_xmin": box[0],
                                            "object_ymin": box[1],
                                            "object_xmax": box[2],
                                            "object_ymax": box[3],
                                            "violation_flag": {
                                                "PD": violation_trackers[index]["PD"][
                                                    "violation"
                                                ],
                                                "SV": violation_trackers[index]["SV"][
                                                    "violation"
                                                ],
                                                "HH": violation_trackers[index]["HH"][
                                                    "violation"
                                                ],
                                            },
                                        },
                                        daemon=True,
                                        name=f"PD logger {count}",
                                    ).start()

                        elif (
                            violation_trackers[index]["PD"]["violation"]
                            and len(person_boxes) == 0
                        ):
                            if violation_trackers[index]["PD"]["end_time"] == None:
                                violation_trackers[index]["PD"][
                                    "end_time"
                                ] = current_time
                                print(
                                    "*" * 100,
                                    "\nViolation ended : Person not wearing hard hat\n",
                                    "*" * 100,
                                )
                            elif current_time - violation_trackers[index]["PD"][
                                "end_time"
                            ] > timedelta(seconds=interval_time):
                                violation_trackers[index]["PD"]["violation"] = False
                                if db_access:
                                    Thread(
                                        target=logging,
                                        args=(
                                            engine,
                                            frame,
                                            violation_trackers[index]["PD"]["end_time"],
                                        ),
                                        kwargs={
                                            "label_id": label_id,
                                            "inference_engine_id": inference_engine_id,
                                            "label_object_pred_threshold": min_score_thresh,
                                            "operating_unit_id": operating_unit_ids[
                                                index
                                            ],
                                            "violation_flag": {
                                                "PD": violation_trackers[index]["PD"][
                                                    "violation"
                                                ],
                                                "SV": violation_trackers[index]["SV"][
                                                    "violation"
                                                ],
                                                "HH": violation_trackers[index]["HH"][
                                                    "violation"
                                                ],
                                            },
                                        },
                                        daemon=True,
                                        name=f"HH logger {count}",
                                    ).start()
                                # cv2.imwrite(f"ppe{count}_pd_no_error.jpg", frame)
                                print(
                                    "*" * 100,
                                    f'\nAfter { {current_time-violation_trackers[index]["PD"]["end_time"]}} seconds Violation ended : Person not wearing hard hat \n',
                                    "*" * 100,
                                )
                                violation_trackers[index]["PD"]["start_time"] = None
                                violation_trackers[index]["PD"]["end_time"] = None

                    if 2 in labels_list[index] and len(person_boxes) > 0:
                        print(
                            "HH",
                            violation_trackers[index]["HH"]["end_time"],
                            violation_trackers[index]["HH"]["start_time"],
                            current_time,
                        )
                        box_mapper = []
                        for person_box_id in range(len(person_boxes)):
                            box_id, flag = check_hardhat(
                                hat_boxes, person_boxes[person_box_id]
                            )
                            person_box = person_boxes[person_box_id]
                            box_mapper.append(
                                {"box_index": box_id, "person_box": person_box}
                            )
                            persons_wearing_hats.append(flag)

                        if (
                            not violation_trackers[index]["HH"]["violation"]
                        ) and not all(persons_wearing_hats):
                            violation_trackers[index]["HH"]["violation"] = True
                            violation_trackers[index]["HH"]["start_time"] = current_time
                            violation_trackers[index]["HH"]["end_time"] = None
                            print(
                                "*" * 100,
                                "\nViolation Detected : Person not wearing hard hat\n",
                                "*" * 100,
                            )
                            for box_map_id, box_map in enumerate(box_mapper):
                                event_flag = 1
                                if db_access:
                                    if box_map["box_index"] == True and box_map_id < len(hat_boxes):
                                        print(box_map, len(hat_boxes))
                                        event_flag = 0
                                        box = hat_boxes[box_map_id]
                                        Thread(
                                            target=logging,
                                            args=(
                                                engine,
                                                frame,
                                                violation_trackers[index]["HH"][
                                                    "start_time"
                                                ],
                                            ),
                                            kwargs={
                                                "label_id": label_id,
                                                "inference_engine_id": inference_engine_id,
                                                "label_object_pred_threshold": min_score_thresh,
                                                "operating_unit_id": operating_unit_ids[
                                                    index
                                                ],
                                                "event_flag": 0,
                                                "index": box_id,
                                                "object_xmin": box[0],
                                                "object_ymin": box[1],
                                                "object_xmax": box[2],
                                                "object_ymax": box[3],
                                                "violation_flag": {
                                                    "PD": violation_trackers[index][
                                                        "PD"
                                                    ]["violation"],
                                                    "SV": violation_trackers[index][
                                                        "SV"
                                                    ]["violation"],
                                                    "HH": violation_trackers[index][
                                                        "HH"
                                                    ]["violation"],
                                                },
                                            },
                                            daemon=True,
                                            name=f"HH logger {count}_hh_no_error",
                                        ).start()

                                    # else:
                                    # cv2.imwrite(f"ppe{count}_hh_error.jpg", frame)

                                    box = box_map["person_box"]
                                    Thread(
                                        target=logging,
                                        args=(
                                            engine,
                                            frame,
                                            violation_trackers[index]["HH"][
                                                "start_time"
                                            ],
                                        ),
                                        kwargs={
                                            "label_id": label_id,
                                            "inference_engine_id": inference_engine_id,
                                            "label_object_pred_threshold": min_score_thresh,
                                            "operating_unit_id": operating_unit_ids[
                                                index
                                            ],
                                            "event_flag": event_flag,
                                            "index": box_id,
                                            "object_xmin": box[0],
                                            "object_ymin": box[1],
                                            "object_xmax": box[2],
                                            "object_ymax": box[3],
                                            "violation_flag": {
                                                "PD": violation_trackers[index]["PD"][
                                                    "violation"
                                                ],
                                                "SV": violation_trackers[index]["SV"][
                                                    "violation"
                                                ],
                                                "HH": violation_trackers[index]["HH"][
                                                    "violation"
                                                ],
                                            },
                                        },
                                        daemon=True,
                                        name=f"HH logger {count}_hh_error",
                                    ).start()

                        elif violation_trackers[index]["HH"]["violation"] and all(
                            persons_wearing_hats
                        ):
                            if violation_trackers[index]["HH"]["end_time"] == None:
                                violation_trackers[index]["HH"][
                                    "end_time"
                                ] = current_time
                                print(
                                    "*" * 100,
                                    "\nViolation ended : Person not wearing hard hat\n",
                                    "*" * 100,
                                )
                            elif current_time - violation_trackers[index]["HH"][
                                "end_time"
                            ] > timedelta(seconds=interval_time):
                                violation_trackers[index]["HH"]["violation"] = False
                                if db_access:
                                    Thread(
                                        target=logging,
                                        args=(
                                            engine,
                                            frame,
                                            violation_trackers[index]["HH"]["end_time"],
                                        ),
                                        kwargs={
                                            "label_id": label_id,
                                            "inference_engine_id": inference_engine_id,
                                            "label_object_pred_threshold": min_score_thresh,
                                            "operating_unit_id": operating_unit_ids[
                                                index
                                            ],
                                            "violation_flag": {
                                                "PD": violation_trackers[index]["PD"][
                                                    "violation"
                                                ],
                                                "SV": violation_trackers[index]["SV"][
                                                    "violation"
                                                ],
                                                "HH": violation_trackers[index]["HH"][
                                                    "violation"
                                                ],
                                            },
                                        },
                                        daemon=True,
                                        name=f"HH logger {count}",
                                    ).start()

                                print(
                                    "*" * 100,
                                    f'\nAfter { {current_time-violation_trackers[index]["HH"]["end_time"]}} seconds Violation ended : Person not wearing hard hat \n',
                                    "*" * 100,
                                )
                                # cv2.imwrite(f"ppe{count}_hh_no_error.jpg", frame)
                                violation_trackers[index]["HH"]["start_time"] = None
                                violation_trackers[index]["HH"]["end_time"] = None

                    elif (
                        2 in labels_list[index]
                        and len(hat_boxes) == len(person_boxes)
                        and violation_trackers[index]["HH"]["violation"]
                    ):
                        label_id = 2
                        if violation_trackers[index]["HH"]["end_time"] != None:
                            if current_time - violation_trackers[index]["HH"][
                                "end_time"
                            ] > timedelta(seconds=interval_time):
                                violation_trackers[index]["HH"]["violation"] = False
                                if db_access:
                                    Thread(
                                        target=logging,
                                        args=(
                                            engine,
                                            frame,
                                            violation_trackers[index]["HH"]["end_time"],
                                        ),
                                        kwargs={
                                            "label_id": label_id,
                                            "inference_engine_id": inference_engine_id,
                                            "label_object_pred_threshold": min_score_thresh,
                                            "operating_unit_id": operating_unit_ids[
                                                index
                                            ],
                                            "violation_flag": {
                                                "PD": violation_trackers[index]["PD"][
                                                    "violation"
                                                ],
                                                "SV": violation_trackers[index]["SV"][
                                                    "violation"
                                                ],
                                                "HH": violation_trackers[index]["HH"][
                                                    "violation"
                                                ],
                                            },
                                        },
                                        daemon=True,
                                        name=f"HH logger {count}",
                                    ).start()

                                # cv2.imwrite(f"ppe{count}_hh_no_error.jpg", frame)
                                print(
                                    "*" * 100,
                                    f'\nAfter { {current_time-violation_trackers[index]["HH"]["end_time"]}} seconds Violation ended : Person not wearing hard hat \n',
                                    "*" * 100,
                                )
                                violation_trackers[index]["HH"]["start_time"] = None
                                violation_trackers[index]["HH"]["end_time"] = None

                    if 3 in labels_list[index] and len(person_boxes) > 0:
                        label_id = 3
                        print(
                            "SV",
                            violation_trackers[index]["SV"]["end_time"],
                            violation_trackers[index]["SV"]["start_time"],
                            current_time,
                        )
                        frame = vis_utils.visualize_boxes_and_labels_on_image_array(
                            frame,
                            output_dict["detection_boxes"],
                            output_dict["detection_classes"],
                            output_dict["detection_scores"],
                            category_index,
                            instance_masks=output_dict.get("detection_masks"),
                            use_normalized_coordinates=True,
                            line_thickness=2,
                        )
                        box_mapper = []
                        for person_box_id in range(len(person_boxes)):
                            box_id, flag = check_vest(
                                vest_boxes, person_boxes[person_box_id]
                            )
                            person_box = person_boxes[person_box_id]
                            box_mapper.append(
                                {"box_index": box_id, "person_box": person_box}
                            )
                            persons_wearing_vests.append(flag)

                        if (
                            not violation_trackers[index]["SV"]["violation"]
                        ) and not all(persons_wearing_vests):
                            violation_trackers[index]["SV"]["violation"] = True
                            violation_trackers[index]["SV"]["start_time"] = current_time
                            violation_trackers[index]["SV"]["end_time"] = None
                            print(
                                "*" * 100,
                                "\nViolation Detected : Person not wearing saftey vest\n",
                                "*" * 100,
                            )
                            for box_map_id, box_map in enumerate(box_mapper):
                                event_flag = 1
                                if db_access:
                                    if box_map["box_index"] == True and box_map_id < len(vest_boxes):

                                        # cv2.imwrite(
                                        #     f"ppe{count}_sv_no_error.jpg", frame
                                        # )
                                        event_flag = 0
                                        box = vest_boxes[box_map_id]
                                        Thread(
                                            target=logging,
                                            args=(
                                                engine,
                                                frame,
                                                violation_trackers[index]["SV"][
                                                    "start_time"
                                                ],
                                            ),
                                            kwargs={
                                                "label_id": label_id,
                                                "inference_engine_id": inference_engine_id,
                                                "label_object_pred_threshold": min_score_thresh,
                                                "operating_unit_id": operating_unit_ids[
                                                    index
                                                ],
                                                "event_flag": 0,
                                                "index": box_id,
                                                "object_xmin": box[0],
                                                "object_ymin": box[1],
                                                "object_xmax": box[2],
                                                "object_ymax": box[3],
                                                "violation_flag": {
                                                    "PD": violation_trackers[index][
                                                        "PD"
                                                    ]["violation"],
                                                    "SV": violation_trackers[index][
                                                        "SV"
                                                    ]["violation"],
                                                    "HH": violation_trackers[index][
                                                        "HH"
                                                    ]["violation"],
                                                },
                                            },
                                            daemon=True,
                                            name=f"sv logger {count}",
                                        ).start()
                                    # else:
                                    # cv2.imwrite(f"ppe{count}_sv_error.jpg", frame)
                                    box = box_map["person_box"]
                                    Thread(
                                        target=logging,
                                        args=(
                                            engine,
                                            frame,
                                            violation_trackers[index]["SV"][
                                                "start_time"
                                            ],
                                        ),
                                        kwargs={
                                            "label_id": label_id,
                                            "inference_engine_id": inference_engine_id,
                                            "label_object_pred_threshold": min_score_thresh,
                                            "operating_unit_id": operating_unit_ids[
                                                index
                                            ],
                                            "event_flag": event_flag,
                                            "index": box_id,
                                            "object_xmin": box[0],
                                            "object_ymin": box[1],
                                            "object_xmax": box[2],
                                            "object_ymax": box[3],
                                            "violation_flag": {
                                                "PD": violation_trackers[index]["PD"][
                                                    "violation"
                                                ],
                                                "SV": violation_trackers[index]["SV"][
                                                    "violation"
                                                ],
                                                "HH": violation_trackers[index]["HH"][
                                                    "violation"
                                                ],
                                            },
                                        },
                                        daemon=True,
                                        name=f"sv logger {count}",
                                    ).start()

                        elif violation_trackers[index]["SV"]["violation"] and all(
                            persons_wearing_vests
                        ):
                            if violation_trackers[index]["SV"]["end_time"] == None:
                                violation_trackers[index]["SV"][
                                    "end_time"
                                ] = current_time
                                print(
                                    "*" * 100,
                                    "\nViolation ended : Person not wearing hard hat\n",
                                    "*" * 100,
                                )
                            elif current_time - violation_trackers[index]["SV"][
                                "end_time"
                            ] > timedelta(seconds=interval_time):
                                violation_trackers[index]["SV"]["violation"] = False
                                # cv2.imwrite(f"ppe{count}_sv_no_error.jpg", frame)
                                if db_access:
                                    Thread(
                                        target=logging,
                                        args=(
                                            engine,
                                            frame,
                                            violation_trackers[index]["SV"]["end_time"],
                                        ),
                                        kwargs={
                                            "label_id": label_id,
                                            "inference_engine_id": inference_engine_id,
                                            "label_object_pred_threshold": min_score_thresh,
                                            "operating_unit_id": operating_unit_ids[
                                                index
                                            ],
                                            "violation_flag": {
                                                "PD": violation_trackers[index]["PD"][
                                                    "violation"
                                                ],
                                                "SV": violation_trackers[index]["SV"][
                                                    "violation"
                                                ],
                                                "HH": violation_trackers[index]["HH"][
                                                    "violation"
                                                ],
                                            },
                                        },
                                        daemon=True,
                                        name=f"sv logger {count}",
                                    ).start()
                                print(
                                    "*" * 100,
                                    f'\nAfter { {current_time-violation_trackers[index]["SV"]["end_time"]}} seconds Violation ended : Person not wearing hard hat \n',
                                    "*" * 100,
                                )
                                # cv2.imwrite(f"ppe{count}_sv_no_error.jpg", frame)
                                violation_trackers[index]["SV"]["start_time"] = None
                                violation_trackers[index]["SV"]["end_time"] = None

                    elif (
                        3 in labels_list[index]
                        and len(vest_boxes) == len(person_boxes)
                        and violation_trackers[index]["SV"]["violation"]
                    ):
                        label_id = 3
                        if violation_trackers[index]["SV"]["end_time"] != None:
                            if current_time - violation_trackers[index]["SV"][
                                "end_time"
                            ] > timedelta(seconds=interval_time):
                                violation_trackers[index]["SV"]["violation"] = False
                                # cv2.imwrite(f"ppe{count}_sv_no_error.jpg", frame)
                                if db_access:
                                    Thread(
                                        target=logging,
                                        args=(
                                            engine,
                                            frame,
                                            violation_trackers[index]["SV"]["end_time"],
                                        ),
                                        kwargs={
                                            "label_id": label_id,
                                            "inference_engine_id": inference_engine_id,
                                            "label_object_pred_threshold": min_score_thresh,
                                            "operating_unit_id": operating_unit_ids[
                                                index
                                            ],
                                            "violation_flag": {
                                                "PD": violation_trackers[index]["PD"][
                                                    "violation"
                                                ],
                                                "SV": violation_trackers[index]["SV"][
                                                    "violation"
                                                ],
                                                "HH": violation_trackers[index]["HH"][
                                                    "violation"
                                                ],
                                            },
                                        },
                                        daemon=True,
                                        name=f"sv logger {count}",
                                    ).start()
                                print(
                                    "*" * 100,
                                    f'\nAfter { {current_time-violation_trackers[index]["SV"]["end_time"]}} seconds Violation ended : Person not wearing hard hat \n',
                                    "*" * 100,
                                )
                                violation_trackers[index]["SV"]["start_time"] = None
                                violation_trackers[index]["SV"]["end_time"] = None

                    for vt in violation_trackers:
                        print(
                            count,
                            vt["PD"]["violation"],
                            vt["HH"]["violation"],
                            vt["SV"]["violation"],
                        )

                    del persons_wearing_hats, persons_wearing_vests

        print("predict:", "releasing video capture")
        cap.release()
        cv2.destroyAllWindows()
        workbook.close()


def main():
    parser = argparse.ArgumentParser(
        description="Hardhat and Vest Detection", add_help=True
    )
    parser.add_argument(
        "--video_file_name",
        type=str,
        required=True,
        help="path to video file, or camera device, i.e /dev/video1 or just 0, 1, ...",
    )
    parser.add_argument(
        "--skip_frame_count",
        type=int,
        default=20,
        required=False,
        help="number of frames to be skipped for each stream read",
    )

    parser.add_argument(
        "--interval_time", type=int, required=True, help="Interval of update to db"
    )
    # parser.add_argument("--model_dir", type=str, required=True, help="path to model directory")
    args = parser.parse_args()

    # frozen_model_path = os.path.join(args.model_dir, "frozen_inference_graph.pb")
    frozen_model_path = "frozen_inference_graph.pb"
    if not os.path.exists(frozen_model_path):
        print("frozen_inference_graph.db file is not exist in model directory")
        exit(-1)
    print("loading model")
    graph = load_model(frozen_model_path)

    predict(
        graph,
        [[1, 2, 3]],
        interval_time=args.interval_time,
        streams={1: args.video_file_name},
        skip_frame_count=args.skip_frame_count
    )


if __name__ == "__main__":
    main()
