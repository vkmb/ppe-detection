from db_cam_helper import *
import cv2

engine = generate_db_engine(creds)

# get confifuration
seq, operating_unit_id, inference_engine_id, label_id = ou_inference_loader(engine)
# load label meta data
label_dict = label_loader(engine, label_id)
# model metadata 
inference_engine_dict = inference_engine_loader(engine, inference_engine_id)
# load operating unit metadata
operating_unit_dict = operating_unit_loader(engine, operating_unit_id)
# run model for the given label , op unit123 

# ['seq', 'id', 'frame_name', 'frame_stored_location', 'frame_size', 'latitude', 'longitude', 'frame_stored_encoding', 'frame_local_epoch', 'frame_local_timestamp', 'frame_local_time_zone', 'created_date', 'created_by', 'active_flag', 'current_flag', 'delete_flag']

def logging(image, timestamp, label_id, inference_engine_id, operating_unit_id, current_flag=1, active_flag=1, delete_flag=0, event_flag=0, object_xmin=0, object_ymin=0, object_xmax=0, object_ymax=0, label_object_pred_threshold=0, label_object_pred_confidence=0 ):
    frame_dict, object_dtl_dict = {}, {}
    time = timestamp.strftime("%d/%M/%Y %I:%M:%S %p")
    file_name = f'{operating_unit_id} {inference_engine_id} {label_id} {time}'
    cv2.imwrite(file_name+".jpg", image)
    frame_dict['frame_name'] = file_name
    frame_dict['frame_stored_location'] = os.path.abspath(file_name+".jpg")
    frame_dict['frame_stored_encoding'] = "JPG"
    frame_dict['frame_local_timestamp'] = f"\'{time}\'"
    frame_dict['frame_local_time_zone'] = 'IST'
    frame_dict['frame_size'] = image.shape
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
    


# ['seq', 'video_id', 'inference_engine_id', 'operating_unit_id', 'label_id', 'frame_id', 'event_processed_local_time', 'event_processed_time_zone', 'event_processed_epoch', 'event_flag', 'created_date', 'created_by', 'update_date', 'updated_by', 'active_flag', 'current_flag', 'delete_flag', 'operating_unit_seq', 'label_seq', 'video_dtl_seq']