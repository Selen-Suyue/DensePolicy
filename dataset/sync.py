from __future__ import annotations

import os
import numpy as np
from datetime import datetime, timedelta

from .motive_util import (
    load_mocap_csv_timestamp as _load_mocap_csv_timestamp,
    load_mocap_csv as _load_mocap_csv,
)


def millisec_to_timestamp(intv: int):
    try:
        intv = int(intv) / 1000
        return datetime.fromtimestamp(intv)
    except (OSError, ValueError):
        return None


def timestamp_to_millisec(timestamp: datetime):
    return int(timestamp.timestamp() * 1000)


def load_teleop_stream_timestamp(teleop_stream_filedir: str):
    teleop_stream_timestamp_list = []
    for el in os.listdir(teleop_stream_filedir):
        ext_split = os.path.splitext(el)
        if ext_split[-1] not in [".png", ".npy", ".pkl"]:
            continue

        ts = millisec_to_timestamp(ext_split[0])
        if ts is None:
            continue

        teleop_stream_timestamp_list.append(ts)

    # sort the list
    teleop_stream_timestamp_list.sort()

    return teleop_stream_timestamp_list


# load_mocap_csv_timestamp
## just use the import version
def load_mocap_csv_timestamp(csvfile):
    return _load_mocap_csv_timestamp(csvfile)


def load_mocap_csv(csvfile, timestamp_list=None):
    if timestamp_list is None:
        timestamp_list = load_mocap_csv_timestamp(csvfile)
    mocap_posemap = _load_mocap_csv(csvfile)

    res = {}
    for _obj_id, _obj_posevec_map in mocap_posemap.items():
        for _ts, (_fid, _obj_posevec) in zip(timestamp_list, _obj_posevec_map.items()):
            if _ts not in res:
                res[_ts] = {}
            res[_ts][_obj_id] = _obj_posevec
    return res


def interval_intersection(int_a: datetime, int_b: datetime):
    return (
        max(int_a[0], int_b[0]),
        min(int_a[1], int_b[1]),
    )


def even_clock(
    record_interval,
    sync_fps: float = 10.0,
) -> list[datetime]:
    record_begin = record_interval[0]
    record_end = record_interval[1]

    timedelta_sec = 1.0 / sync_fps
    timedelta_ = timedelta(seconds=timedelta_sec)
    timepoint_current = record_begin

    stream_timestamp_list = []
    while timepoint_current <= record_end:
        stream_timestamp_list.append(timepoint_current)
        timepoint_current = timepoint_current + timedelta_
    return stream_timestamp_list


def sync_stream(
    multi_stream_timestamp: dict[str, list[datetime]],
    ref_stream_timestamp: list[datetime],
):
    synced_stream_timestamp = {}
    syncing_current_offset = {}
    for stream_name in multi_stream_timestamp.keys():
        synced_stream_timestamp[stream_name] = []
        syncing_current_offset[stream_name] = 0

    for timepoint_current in ref_stream_timestamp:
        for stream_name, stream_timestamp_list in multi_stream_timestamp.items():
            _scan_offset = syncing_current_offset[stream_name]
            while (_scan_offset < len(stream_timestamp_list) and
                   stream_timestamp_list[_scan_offset] <= timepoint_current):
                _scan_offset += 1
            ## compare _scan_offset and max(_scan_offset - 1, 0), select a closer
            _scan_offset_prev = max(_scan_offset - 1, 0)
            if (_scan_offset
                    == len(stream_timestamp_list)) or (timepoint_current - stream_timestamp_list[_scan_offset_prev]
                                                       <= stream_timestamp_list[_scan_offset] - timepoint_current):
                syncing_current_offset[stream_name] = _scan_offset_prev
            else:
                syncing_current_offset[stream_name] = _scan_offset
            # append timestamp to synced_stream_timestamp using current offset
            synced_stream_timestamp[stream_name].append(stream_timestamp_list[syncing_current_offset[stream_name]])

    return synced_stream_timestamp
