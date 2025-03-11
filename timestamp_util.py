import time
import datetime
import os

global_timestamp = time.time()


def ts_to_str(ts_in):
    return datetime.datetime.fromtimestamp(ts_in).strftime("%Y-%m-%d-%H-%M-%S")


global_timestamp_str = ts_to_str(global_timestamp)


def str_to_ts(str_in):
    return datetime.datetime.strptime(str_in, "%Y-%m-%d-%H-%M-%S").timestamp()


def is_newer(f1, f2):
    if isinstance(f1, str):
        f1 = os.path.getmtime(f1)
    if isinstance(f2, str):
        f2 = os.path.getmtime(f2)

    return f1 > f2

