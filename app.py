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