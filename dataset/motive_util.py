import csv
import re
import numpy as np
from datetime import datetime, timedelta


point_matcher = re.compile(r"^Point (\d+)$")


def load_mocap_object_asset(asset_path):
    # input like this:
    # File Type,Rigid Body Asset
    # Version,1
    # Name,O02@0010@00003
    # ID,12836801337049616877
    # Color,0,235,255
    # Units,m
    # Point,Point 1,0.027612,0.017955,-0.045468,None
    # Point,Point 2,-0.030078,-0.021164,-0.032535,None
    # Point,Point 3,-0.018698,-0.028876,0.038051,None
    # Point,Point 4,0.021164,0.032085,0.039953,None
    #
    # output a numpy array of shape (N, 3) with each row being a point
    with open(asset_path, "r") as f:
        reader = csv.reader(f)
        point_list = []
        for line in reader:
            if line[0] != "Point":
                continue

            # point_id = int(point_matcher.match(line[1]).group(1))

            point = np.array(line[2:5], dtype=np.float32)
            point_list.append(point)

    point_arr = np.stack(point_list, axis=0)
    return point_arr


def triangulate_mocap_object_asset(asset_pc):
    from scipy.spatial import Delaunay

    if len(asset_pc) == 3:
        line_arr = np.array(
            [[0, 1], [1, 2], [2, 0]],
            dtype=np.int32,
        )
        return line_arr

    # input a numpy array of shape (N, 3) with each row being a point
    # output a numpy array of shape (M, 3) with each row being a line
    tri = Delaunay(asset_pc)
    smplices = tri.simplices
    # for each simplice, append the lines (represented in point index) to the list
    line_set = set()
    for smp in smplices:
        # each simplex should have 6 edges, with lower index first
        line_set.add((smp[0], smp[1]))
        line_set.add((smp[0], smp[2]))
        line_set.add((smp[0], smp[3]))
        line_set.add((smp[1], smp[2]))
        line_set.add((smp[1], smp[3]))
        line_set.add((smp[2], smp[3]))
    line_arr = np.array(list(line_set), dtype=np.int32)
    return line_arr


def load_mocap_csv(csvfile):
    with open(csvfile, "r") as file:
        csv_reader = csv.reader(file)

        csv_reader_iter = iter(csv_reader)

        header_row = next(csv_reader_iter)
        header_info = {}
        # iterate header_row each 2 elements
        for _i in range(0, len(header_row), 2):
            header_info[header_row[_i]] = header_row[_i + 1]
        if header_info["Length Units"] == "Millimeters":
            tsl_divisor = 1000.0
        else:
            tsl_divisor = 1.0

        next(csv_reader_iter)  # skip padding

        type_row = next(csv_reader_iter)
        name_row = next(csv_reader_iter)
        iden_row = next(csv_reader_iter)
        attr_row = next(csv_reader_iter)
        meta_row = next(csv_reader_iter)
        # for each row index, get a dict of the type name attr meta
        desc_list = []
        for _t, _n, _a, _m in zip(type_row, name_row, attr_row, meta_row):
            desc_list.append(
                {
                    "type": _t,
                    "name": _n,
                    "attr": _a,
                    "meta": _m,
                }
            )

        # iterate over all desc, find a list of object id
        obj_id_set = set()
        obj_posevec_index_map = {}
        for _col_offset, _desc in enumerate(desc_list):
            if _desc["type"] == "Rigid Body":
                name = _desc["name"]
                obj_id_set.add(name)

                if name not in obj_posevec_index_map:
                    obj_posevec_index_map[name] = {}
                _handle = obj_posevec_index_map[name]

                attr = _desc["attr"]
                if attr == "Rotation":
                    meta = _desc["meta"]
                    if meta == "X":
                        _handle["qx"] = _col_offset
                    elif meta == "Y":
                        _handle["qy"] = _col_offset
                    elif meta == "Z":
                        _handle["qz"] = _col_offset
                    elif meta == "W":
                        _handle["qw"] = _col_offset
                elif attr == "Position":
                    meta = _desc["meta"]
                    if meta == "X":
                        _handle["x"] = _col_offset
                    elif meta == "Y":
                        _handle["y"] = _col_offset
                    elif meta == "Z":
                        _handle["z"] = _col_offset

        obj_id_list = sorted(obj_id_set)

        # iterate over all rows, find the posevec for each object
        res = {_o: {} for _o in obj_id_list}
        for row in csv_reader_iter:
            frame_id = int(row[0])
            for _obj_id in obj_id_list:
                _handle = obj_posevec_index_map[_obj_id]
                _x = float(row[_handle["x"]]) / tsl_divisor
                _y = float(row[_handle["y"]]) / tsl_divisor
                _z = float(row[_handle["z"]]) / tsl_divisor
                _qw = float(row[_handle["qw"]])
                _qx = float(row[_handle["qx"]])
                _qy = float(row[_handle["qy"]])
                _qz = float(row[_handle["qz"]])
                _posevec = np.array([_x, _y, _z, _qw, _qx, _qy, _qz], dtype=np.float32)
                res[_obj_id][frame_id] = _posevec

    return res


def load_mocap_csv_marker(csvfile):
    with open(csvfile, "r") as file:
        csv_reader = csv.reader(file)

        csv_reader_iter = iter(csv_reader)

        header_row = next(csv_reader_iter)
        header_info = {}
        # iterate header_row each 2 elements
        for _i in range(0, len(header_row), 2):
            header_info[header_row[_i]] = header_row[_i + 1]
        if header_info["Length Units"] == "Millimeters":
            tsl_divisor = 1000.0
        else:
            tsl_divisor = 1.0

        next(csv_reader_iter)  # skip padding

        type_row = next(csv_reader_iter)
        name_row = next(csv_reader_iter)
        iden_row = next(csv_reader_iter)
        attr_row = next(csv_reader_iter)
        meta_row = next(csv_reader_iter)
        # for each row index, get a dict of the type name attr meta
        desc_list = []
        for _t, _n, _a, _m in zip(type_row, name_row, attr_row, meta_row):
            desc_list.append(
                {
                    "type": _t,
                    "name": _n,
                    "attr": _a,
                    "meta": _m,
                }
            )

        # iterate over all desc, find a list of object id
        marker_id_set = set()
        marker_pos_index_map = {}
        for _col_offset, _desc in enumerate(desc_list):
            if _desc["type"] == "Marker":
                name = _desc["name"]

                if name.startswith("Unlabeled") or name.startswith("FKA"):
                    continue

                marker_id_set.add(name)

                if name not in marker_pos_index_map:
                    marker_pos_index_map[name] = {}
                _handle = marker_pos_index_map[name]

                attr = _desc["attr"]
                if attr == "Position":
                    meta = _desc["meta"]

                    if meta == "X":
                        _handle["x"] = _col_offset
                    elif meta == "Y":
                        _handle["y"] = _col_offset
                    elif meta == "Z":
                        _handle["z"] = _col_offset
        marker_id_list = sorted(marker_id_set)

        # iterate over all rows, find the pos for each marker
        res = {}
        for row in csv_reader_iter:
            frame_id = int(row[0])
            res[frame_id] = {}
            for _m_id in marker_id_list:
                _handle = marker_pos_index_map[_m_id]
                try:
                    _x = float(row[_handle["x"]]) / tsl_divisor
                    _y = float(row[_handle["y"]]) / tsl_divisor
                    _z = float(row[_handle["z"]]) / tsl_divisor
                    _tsl = np.array([_x, _y, _z, 1.0], dtype=np.float32)
                except ValueError:
                    _tsl = np.zeros((4,), dtype=np.float32)
                res[frame_id][_m_id] = _tsl

    # return res
    return res


def motive_timestamp_convert(date_str):
    # Replace Chinese AM/PM notation with English AM/PM
    if "下午" in date_str:
        date_str = date_str.replace("下午", "PM")
    elif "上午" in date_str:
        date_str = date_str.replace("上午", "AM")

    # Define the format matching your date string
    date_format = "%Y-%m-%d %I.%M.%S.%f %p"

    # Parse the date string into a datetime object
    dt = datetime.strptime(date_str, date_format)

    return dt


def load_mocap_csv_timestamp(csvfile):
    res = []
    with open(csvfile, "r") as file:
        csv_reader = csv.reader(file)

        csv_reader_iter = iter(csv_reader)

        header_row = next(csv_reader_iter)
        header_info = {}
        # iterate header_row each 2 elements
        for _i in range(0, len(header_row), 2):
            header_info[header_row[_i]] = header_row[_i + 1]
        if header_info["Length Units"] == "Millimeters":
            tsl_divisor = 1000.0
        else:
            tsl_divisor = 1.0

        next(csv_reader_iter)  # skip padding
        type_row = next(csv_reader_iter)
        name_row = next(csv_reader_iter)
        iden_row = next(csv_reader_iter)
        attr_row = next(csv_reader_iter)
        meta_row = next(csv_reader_iter)
        # for each row index, get a dict of the type name attr meta
        desc_list = []
        for _t, _n, _a, _m in zip(type_row, name_row, attr_row, meta_row):
            desc_list.append(
                {
                    "type": _t,
                    "name": _n,
                    "attr": _a,
                    "meta": _m,
                }
            )
        assert desc_list[1]["meta"] == "Time (Seconds)"

        frame_rate = float(header_info["Export Frame Rate"])
        capture_start_time = header_info["Capture Start Time"]
        capture_start_time = motive_timestamp_convert(capture_start_time)

        for row in csv_reader_iter:
            _timedelta_sec = float(row[1])
            timedelta_sec = timedelta(seconds=_timedelta_sec)
            frame_timestemp = capture_start_time + timedelta_sec
            res.append(frame_timestemp)

    return res
