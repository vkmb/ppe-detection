# pip install sqlalchemy psycopg2-binary
# sudo apt-get install build-dep python-psycopg2
# pip install --upgrade onvif_zeep sqlalchemy psycopg2-binary

import os
import cv2
import sys
import psycopg2
import xlsxwriter
import sqlalchemy as db
from datetime import datetime
from urllib.parse import quote
from threading import Thread
from onvif import ONVIFCamera


# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook(f"test_report_{datetime.now()}.xlsx")
worksheet = workbook.add_worksheet()
headers = [
    "Violation Type",
    "Image",
    "timestamp",
    "object_dtl_seq",
    "frame_seq",
    "event_log_seq",
    "image_file_path"
]
row = 0
worksheet.set_column("A:G", 100)
worksheet.set_column("B:B", 200)
for datum in headers:
    worksheet.write(row, headers.index(datum), datum)


def test_log(data):
    # data - list [type of violation, image_file_path, image_file_path, timestamp, object_dtl_seq, frame_seq, event_log_seq]
    global row
    row += 1
    worksheet.set_row(row, 200)
    for datum in data:
        if data.index(datum) == 1:
            worksheet.insert_image(
                row,
                data.index(datum),
                datum,
                {"x_scale": 0.5, "y_scale": 0.5, "x_offset": 15, "y_offset": 15},
            )
            continue
        worksheet.write(row, data.index(datum), datum)


def list_tables(engine):
    if engine == None:
        return None
    table_list = []
    with engine.connect() as link_to_db:
        query = f"SELECT table_name FROM information_schema.tables WHERE table_schema='public' or table_schema='CV_Analytics'"
        result = link_to_db.execute(query)
        table_list = list(map(lambda x: x.values()[-1], result.fetchall()))
    return table_list


def dict_generator(column_name_list, column_value_list):
    temp = {}
    if len(column_name_list) == len(column_value_list):
        for column_name, column_value in zip(column_name_list, column_value_list):
            temp[column_name] = column_value
    return temp


def get_table_info(engine, table_name):
    if engine == None or table_name == "":
        return None
    column_meta = []
    with engine.connect() as link_to_db:
        if "." in table_name:
            table_name = table_name.split(".")[-1]
        # query = f'select column_name, data_type, character_maximum_length from INFORMATION_SCHEMA.COLUMNS where table_name =\'{table_name}\''
        query = f"select column_name from INFORMATION_SCHEMA.COLUMNS where table_name ='{table_name}'"
        result = link_to_db.execute(query)
        column_meta = list(map(lambda x: x.values()[-1], result.fetchall()))
    return column_meta


def get_seq(engine, table_name):
    # this can be used for inserting only
    if engine == None or table_name == "":
        return None
    seq = 0
    with engine.connect() as link_to_db:
        query = f"SELECT max(seq) FROM {table_name}"
        result = link_to_db.execute(query)
        if result.rowcount > 0:
            rowproxy = result.fetchall()[0]
            value = rowproxy.values()[0]
            if value == None:
                seq += 1
            else:
                seq = value + 1
        else:
            seq += 1
    return seq


def get_camera_stream_uri(
    ip, usr="admin", psk="password", wsdl_location="/etc/onvif/wsdl/"
):
    # https://www.onvif.org/onvif/ver10/media/wsdl/media.wsdl#op.GetStreamUri
    cam_control = ONVIFCamera(ip, 80, usr, psk, wsdl_location)
    media_control = cam_control.create_media_service()
    cam_config = media_control.GetProfiles()[0].token
    stream_uri_obj = media_control.create_type("GetStreamUri")
    stream_uri_obj.ProfileToken = cam_config
    stream_uri_obj.StreamSetup = {
        "Stream": "RTP-Unicast",
        "Transport": {"Protocol": "RTSP"},
    }
    connection_data = media_control.ws_client.GetStreamUri(
        stream_uri_obj.StreamSetup, stream_uri_obj.ProfileToken
    )
    return connection_data


def event_log_dtl_writer(engine, data_dict):
    table_name = '"CV_Analytics".event_log_dtl'
    seq = -1
    if engine == None or data_dict == {}:
        return None
    with engine.connect() as link_to_db:
        seq = get_seq(engine, table_name)
        data_dict["seq"] = seq
        insert_query = f"INSERT INTO {table_name} ({','.join(data_dict.keys())}) VALUES {tuple(data_dict.values())}"
        link_to_db.execute(insert_query)
    return seq


def label_writer(engine, label_dict, label_template):
    # label_loader(engine, label_dict, label_template)
    table_name = '"CV_Analytics".label'
    if engine == None or label_dict == {}:
        return None
    with engine.connect() as link_to_db:
        for key in label_dict:
            seq = None
            new = True
            q = link_to_db.execute(f"SELECT max(seq) from {table_name}")
            if q.rowcount > 0:
                q2 = link_to_db.execute(
                    f"SELECT * from {table_name} WHERE created_by={label_template['created_by']}"
                )
                rows = q2.fetchall()
                for row in rows:
                    if key in row.values():
                        new = False
                        break
                if new == False:
                    continue
                seq = q.fetchall()[0].values()[0] + 1
            else:
                seq = list(label_dict[key].keys())[0]
            label_template["seq"] = seq
            label_template["id"] = seq
            label_template["label_code"] = key
            label_template["label_name"] = label_dict[key][
                list(label_dict[key].keys())[0]
            ]
            label_template["created_date"] = str(datetime.now())
            insert_query = f"INSERT INTO {table_name} ({','.join(label_template.keys())}) VALUES {tuple(label_template.values())}"
            link_to_db.execute(insert_query)


def frame_writer(engine, frame_dict):
    table_name = '"CV_Analytics".frame'
    if engine == None or frame_dict == None:
        return None
    seq = None
    with engine.connect() as link_to_db:
        seq = get_seq(engine, table_name)
        frame_dict["seq"], frame_dict["id"] = seq, seq
        insert_query = f"INSERT INTO {table_name} ({','.join(frame_dict.keys())}) VALUES {tuple(frame_dict.values())}"
        link_to_db.execute(insert_query)
    return seq


def object_dtl_writer(engine, object_dtl_dict):
    table_name = '"CV_Analytics".object_dtl'
    if engine == None or object_dtl_dict == None:
        return None
    seq = None
    with engine.connect() as link_to_db:
        seq = get_seq(engine, table_name)
        object_dtl_dict["seq"], object_dtl_dict["id"] = seq, seq
        insert_query = f"INSERT INTO {table_name} ({','.join(object_dtl_dict.keys())}) VALUES {tuple(object_dtl_dict.values())}"
        link_to_db.execute(insert_query)
    return seq


def label_loader(engine, label_id, status=1):
    # label_loader(engine, label_dict, label_template)
    table_name = '"CV_Analytics".label'
    label_dict = {}
    if engine == None:
        return None
    with engine.connect() as link_to_db:
        result = link_to_db.execute(
            f"UPDATE {table_name} SET current_flag={status-1}, active_flag={status-1}, updated_date='{datetime.now().replace(tzinfo=None)}' WHERE seq!={label_id}"
        )
        del result
        result = link_to_db.execute(
            f"UPDATE {table_name} SET current_flag={status}, active_flag={status}, updated_date='{datetime.now().replace(tzinfo=None)}' WHERE seq={label_id}"
        )
        if result.rowcount > 0:
            result = link_to_db.execute(
                f"SELECT * from {table_name} WHERE seq={label_id}"
            )
            rowproxy = result.fetchone()
            label_dict = dict_generator(
                get_table_info(engine, table_name), rowproxy.values()
            )
    return label_dict


def inference_engine_writer(engine, inference_engine_dict):
    table_name = '"CV_Analytics".inference_engine'
    if engine == None or inference_engine_dict == {}:
        return None
    with engine.connect() as link_to_db:
        seq = None
        q = link_to_db.execute(f"SELECT max(seq) from {table_name}")
        if q.rowcount > 0:
            seq = q.fetchall()[0].values()[0] + 1
        else:
            seq = 1
        inference_engine_dict["seq"] = seq
        inference_engine_dict["id"] = seq
        inference_engine_dict["created_date"] = str(datetime.now())
        query = f"INSERT INTO {table_name} ({','.join(inference_engine_dict.keys())}) VALUES {tuple(inference_engine_dict.values())}"
        link_to_db.execute(query)


def inference_engine_loader(engine, inference_engine_id, status=1):
    table_name = '"CV_Analytics".inference_engine'
    inference_engine_dict = {}
    if engine == None:
        return None
    with engine.connect() as link_to_db:
        result = link_to_db.execute(
            f"UPDATE {table_name} SET current_flag={status-1}, active_flag={status-1}, updated_date='{datetime.now().replace(tzinfo=None)}' WHERE seq!={inference_engine_id}"
        )
        del result
        result = link_to_db.execute(
            f"UPDATE {table_name} SET current_flag={status}, active_flag={status}, updated_date='{datetime.now().replace(tzinfo=None)}' WHERE seq={inference_engine_id}"
        )
        if result.rowcount > 0:
            result = link_to_db.execute(
                f"SELECT * from {table_name} WHERE seq={inference_engine_id}"
            )
            rowproxy = result.fetchone()
            inference_engine_dict = dict_generator(
                get_table_info(engine, table_name), rowproxy.values()
            )
    return inference_engine_dict


def operating_unit_writer(engine, inference_engine_dict):
    table_name = "operating_unit"
    if engine == None or inference_engine_dict == {}:
        return None
    with engine.connect() as link_to_db:
        seq = None
        q = link_to_db.execute(f"SELECT max(seq) from {table_name}")
        if q.rowcount > 0:
            seq = q.fetchall()[0].values()[0] + 1
        else:
            seq = 1
        inference_engine_dict["seq"] = seq
        inference_engine_dict["id"] = seq
        inference_engine_dict["created_date"] = str(datetime.now())
        query = f"INSERT INTO {table_name} ({','.join(inference_engine_dict.keys())}) VALUES {tuple(inference_engine_dict.values())}"
        link_to_db.execute(query)


def operating_unit_loader(engine, operating_unit_id, status=1):
    table_name = "operating_unit"
    operating_unit_dict = {}
    if engine == None:
        return None
    with engine.connect() as link_to_db:
        result = link_to_db.execute(
            f"UPDATE {table_name} SET current_flag={status-1}, active_flag={status-1}, updated_date='{datetime.now().replace(tzinfo=None)}' WHERE seq!={operating_unit_id}"
        )
        del result
        result = link_to_db.execute(
            f"UPDATE {table_name} SET current_flag={status}, active_flag={status}, updated_date='{datetime.now().replace(tzinfo=None)}' WHERE seq={operating_unit_id}"
        )
        if result.rowcount > 0:
            result = link_to_db.execute(
                f"SELECT * from {table_name} WHERE seq={operating_unit_id}"
            )
            rowproxy = result.fetchone()
            operating_unit_dict = dict_generator(
                get_table_info(engine, table_name), rowproxy.values()
            )
    return operating_unit_dict


def table_generic_writer(engine, table_name, data_dict):
    if engine == None or table_name == "" or data_dict == {}:
        return None
    with engine.connect() as link_to_db:
        seq = get_seq(engine, table_name)
        data_dict["seq"], data_dict["id"] = seq, seq
        insert_query = f"INSERT INTO {table_name} ({','.join(data_dict.keys())}) VALUES {tuple(data_dict.values())}"
        link_to_db.execute(insert_query)


def ou_inference_loader(engine, inference_engine_id=None):
    seq, operating_unit_id, inference_engine_id, label_id = None, None, None, None
    table_name = '"CV_Analytics".ou_inference_engine_label'
    if engine == None:
        return None
    with engine.connect() as link_to_db:
        query = f"SELECT * FROM {table_name} WHERE current_flag=1 and active_flag=1"
        if inference_engine_id != None:
            query += f" and inference_engine_id={inference_engine_id}"
        result = link_to_db.execute(query)
        if result.rowcount == 0:
            print("No configurations found ! exiting !")
            exit()
        else:
            data_list = list(map(lambda x: x.values(), result.fetchall()))
            seq = [record[0] for record in data_list]
            operating_unit_id = [record[1] for record in data_list]
            inference_engine_id = [record[2] for record in data_list]
            label_id = [record[3] for record in data_list]
            temp = {}
            for ieid in inference_engine_id:
                temp.update({ieid: {}})
            for _inference_engine_id, _label_id, _operating_unit_id in zip(
                inference_engine_id, label_id, operating_unit_id
            ):
                if _operating_unit_id not in temp[_inference_engine_id].keys():
                    temp[_inference_engine_id][_operating_unit_id] = [_label_id]
                else:
                    temp[_inference_engine_id][_operating_unit_id].append(_label_id)
            print(
                f"Configuration loaded with the following parameters\n seqenece numbers = {' | '.join(map(str, seq))}\n operating_unit_ids = {' | '.join(map(str, operating_unit_id))}\n inference_engine_ids = {' | '.join(map(str, inference_engine_id))}\n label_ids\t = {' | '.join(map(str, label_id))}\n"
            )
    return (seq, operating_unit_id, inference_engine_id, label_id, temp)


def generate_db_engine(creds):
    if creds == {}:
        return None
    engine = db.create_engine(
        f'postgresql://{creds["usr"]}:{quote(creds["psk"])}@{creds["ipp"]}/{creds["dbn"]}'
    )
    return engine


def logging(
    engine,
    image,
    timestamp,
    label_id,
    inference_engine_id,
    operating_unit_id,
    event_flag=0,
    index="",
    current_flag=1,
    active_flag=1,
    delete_flag=0,
    object_xmin=0,
    object_ymin=0,
    object_xmax=0,
    object_ymax=0,
    label_object_pred_threshold=0,
    label_object_pred_confidence=0,
    violation_flag={"PD": False, "HH": False, "SV": False},
):
    vilotation_type = ""
    if violation_flag["PD"]:
        vilotation_type += "Illegal entry started_"
    else:
        vilotation_type += "Illegal entry ended_"
    if violation_flag["HH"]:
        vilotation_type += "Hard hat started_"
    else:
        vilotation_type += "Hard hat ended_"
    if violation_flag["SV"]:
        vilotation_type += "Saftey Vest started_"
    else:
        vilotation_type += "Saftey Vest ended"
    frame_dict, object_dtl_dict = {}, {}
    time = timestamp.strftime("%m/%d/%Y %I:%M:%S %p")
    time2 = timestamp.strftime("%d%m%Y%_I%M%S_%p")
    object_dtl_id = -1
    file_name = f"{operating_unit_id}_{inference_engine_id}_{label_id}_{time2}_{index}_{vilotation_type}"
    cv2.imwrite(file_name + ".jpg", image)
    shape = ""
    for i in image.shape:
        shape += str(i) + " "
    frame_dict["frame_name"] = file_name
    frame_dict["frame_stored_location"] = os.path.abspath(file_name + ".jpg")
    frame_dict["frame_stored_encoding"] = "JPG"
    frame_dict["frame_local_timestamp"] = f'"{time}"'
    frame_dict["frame_local_time_zone"] = "IST"
    frame_dict["frame_size"] = shape
    frame_dict["created_by"] = inference_engine_id
    frame_dict["created_date"] = f'"{time}"'
    frame_dict["active_flag"] = active_flag
    frame_dict["current_flag"] = current_flag
    frame_dict["delete_flag"] = delete_flag
    frame_id = frame_writer(engine, frame_dict)

    if frame_id == None:
        return None

    if object_xmin != 0 or object_ymin != 0 or object_xmax != 0 or object_ymax != 0:
        object_dtl_dict["frame_id"] = frame_id
        object_dtl_dict["object_loc_id"] = operating_unit_id
        object_dtl_dict["label_id"] = label_id
        object_dtl_dict["created_by"] = inference_engine_id
        object_dtl_dict["created_date"] = f'"{time}"'
        object_dtl_dict["active_flag"] = active_flag
        object_dtl_dict["current_flag"] = current_flag
        object_dtl_dict["delete_flag"] = delete_flag
        object_dtl_dict["object_xmin"] = object_xmin
        object_dtl_dict["object_ymin"] = object_ymin
        object_dtl_dict["object_xmax"] = object_xmax
        object_dtl_dict["object_ymax"] = object_ymax
        object_dtl_dict["label_object_pred_threshold"] = label_object_pred_threshold
        object_dtl_dict["label_object_pred_confidence"] = label_object_pred_confidence
        object_dtl_id = object_dtl_writer(engine, object_dtl_dict)

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
    data_dict["event_processed_local_time"] = f'"{time}"'
    data_dict["event_flag"] = event_flag
    data_dict["created_date"] = f'"{time}"'
    data_dict["created_by"] = inference_engine_id
    data_dict["current_flag"] = current_flag
    data_dict["active_flag"] = active_flag
    data_dict["delete_flag"] = delete_flag

    event_id = event_log_dtl_writer(engine, data_dict)
    # type of violation, image_file_path, object_dtl_seq, frame_seq, event_log_seq

    test_log(
        [
            vilotation_type,
            file_name + ".jpg",
            timestamp,
            os.path.abspath(file_name + ".jpg"),
            object_dtl_id,
            frame_id,
            event_id,
        ]
    )


model_id, current_flag, active_flag, delete_flag = 11, 1, 1, 0
model_config_name = "config.json"
user_id = os.environ["user_id"]

label_template = {
    "seq": None,
    "id": None,
    "label_code": None,
    "label_name": None,
    "created_by": model_id,
    "current_flag": current_flag,
    "active_flag": active_flag,
    "delete_flag": delete_flag,
}

label_dict = {
    "NVL": {5: "No Violation"},
    "VLO": {6: "Violation"},
}


inference_engine_dict = {
    "model_id": model_id,
    "model_vrsn_number": 1,
    "model_name": "PPE Detect",
    "model_path": os.path.abspath(sys.argv[0]),
    "backbone_name": "yolo v3",
    "model_weight_format": ".pb, .h5",
    "model_config_name": model_config_name,
    "model_config_path": os.path.abspath(model_config_name),
    "model_preprocess_input_shape": "416, 416",
    "model_framework": "tf",
    "created_by": 11,
    "current_flag": current_flag,
    "active_flag": active_flag,
    "delete_flag": delete_flag,
}

creds = {
    "usr": os.environ["db_usr"],
    "psk": os.environ["db_psk"],
    "dbn": os.environ["db_name"],
    "ipp": os.environ["db_ipp"],  # to be replaced with env variables
}

if __name__ == "__main__":
    print("Entering interactive mode\nTables available")
    engine = generate_db_engine(creds)
    flag = True
    tables = list_tables(engine)
    while flag:
        for i in range(len(tables)):
            print(f"{i} - {tables[i]}")
        table_index = input("Enter the index number of the table to add data : ")
        try:
            if table_index == "q":
                del engine
                flag = False
                continue
            table_index = int(table_index)
            if table_index not in range(0, len(tables)):
                continue
            else:
                columns = get_table_info(engine, tables[table_index])
                template_dict = {}

                for column in columns:
                    if column == "active_flag":
                        template_dict[column] = 1
                    elif column == "delete_flag":
                        template_dict[column] = 0
                    elif column == "created_by":
                        template_dict[column] = user_id
                    elif column == "created_date":
                        template_dict[column] = str(datetime.now())
                    elif (
                        column == "seq"
                        or column == "id"
                        or column == "updated_by"
                        or column == "updated_date"
                    ):
                        continue
                    else:
                        template_dict[column] = input(f"{column}'s value : ")

                print(template_dict)
                confirmation = input(
                    'The above values will be written to the table\n Type "yes" to confirm : '
                )
                if confirmation == "yes":
                    if (
                        tables[table_index] != "operating_unit"
                        or tables[table_index] != "operating_unit_assembly"
                    ):
                        table_generic_writer(
                            engine,
                            f'"CV_Analytics".{tables[table_index]}',
                            template_dict,
                        )
                    else:
                        table_generic_writer(engine, tables[table_index], template_dict)
                else:
                    print("Dropping insert operation")

        except:
            print("Unexpected Error")
            continue
    else:
        exit()
