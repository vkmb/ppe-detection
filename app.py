import os
import cv2
from predict import *
from db_cam_helper import *

engine = generate_db_engine(creds)

# get confifuration
seqs, operating_unit_ids, inference_engine_ids, label_ids, configuration_dict = ou_inference_loader(engine)
print(configuration_dict)
for inference_engine_id in configuration_dict:
    # model metadata 
    inference_engine_dict = inference_engine_loader(engine, inference_engine_id)
    # load operating unit metadata
    operating_unit_dict = {}
    labels_list = []
    for ou_id in configuration_dict[inference_engine_id]:
        temp = operating_unit_loader(engine, ou_id)
        # operating_unit_serial_number
        operating_unit_dict.update({ temp['seq']: "rtsp://"+temp['operating_unit_serial_number']})
        labels_list.append(configuration_dict[inference_engine_id][ou_id])
    if inference_engine_id == 1:
        frozen_model_path = "frozen_inference_graph.pb"
        if not os.path.exists(frozen_model_path):
            print("frozen_inference_graph.db file is not exist in model directory")
            exit(-1)
        print("loading model")
        graph = load_model(frozen_model_path)
        # frozen_model_path = model_config_name
        predict(graph, inference_engine_id, labels_list, streams=operating_unit_dict, engine=engine)
    else:
        pass