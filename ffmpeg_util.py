import os
import threading
import numpy as np
import time
import json

import ffmpeg


class VideoOut:

    def __init__(self, out_path, video_shape, fps, input_param=None, output_param=None):
        self.out_path = out_path
        self.video_shape = video_shape
        self.fps = fps
        if input_param is None:
            input_param = {
                "pix_fmt": "rgb24",
            }
        self.input_param = input_param
        if output_param is None:
            output_param = {"pix_fmt": "yuv420p", "vcodec": "ffv1"}
        self.output_param = output_param

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.process = (ffmpeg.input(
            "pipe:",
            format="rawvideo",
            **self.input_param,
            s="{}x{}".format(self.video_shape[0], self.video_shape[1]),
            r=fps,
        ).output(self.out_path, **self.output_param, r=fps).overwrite_output().run_async(pipe_stdin=True, quiet=True))
        self.ts_list = []

    def write(self, img: np.ndarray) -> None:
        self.process.stdin.write(img.tobytes())
        timestamp = int(time.time() * 1000)
        self.ts_list.append(timestamp)

    def close(self):
        self.process.stdin.close()
        self.process.wait()
        del self.process

        # save timestamps, replace outpath with .json and save (append if no ext)
        ts_filename = os.path.splitext(self.out_path)[0] + ".json"
        with open(ts_filename, "w") as ofs:
            json.dump(self.ts_list, ofs, indent=4)

        self.process = self.ts_list = None
