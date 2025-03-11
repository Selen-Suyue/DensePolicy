import numpy as np
import logging
import threading
import pyrealsense2 as rs
import cv2
import time
from multiprocessing import shared_memory


class SharedMemoryManager(object):
    """
    Shared Memory Manager.
    """

    def __init__(self, name, type=0, shape=(1, ), dtype=np.float32):
        """
        Initialization.
        
        Parameters
        ----------
        - name: the name of the shared memory;
        - type: integer in [0, 1];
            * 0: sender;
            * 1: receiver.
        - shape: optional, default: (1,), the array shape.
        - dtype: optional, default: np.float32, the element type of the array.
        """
        super(SharedMemoryManager, self).__init__()
        self.name = name
        self.type = type
        self.shape = shape
        if isinstance(dtype, str):
            dtype = to_dtype(dtype)
        self.dtype = np.dtype(dtype)
        if self.type not in [0, 1]:
            raise AttributeError('Invalid type in shared memory manager.')
        if self.type == 0:
            self.shared_memory = shared_memory.SharedMemory(
                name=self.name,
                create=True,
                size=self.dtype.itemsize * np.prod(self.shape))
            self.buf = np.ndarray(self.shape,
                                  dtype=self.dtype,
                                  buffer=self.shared_memory.buf)
        else:
            self.shared_memory = shared_memory.SharedMemory(name=self.name)

    def execute(self, arr=None):
        """
        Execute the function.

        Paramters
        ---------
        - arr: np.array object, only used in sender, the array.
        """
        if self.type == 0:
            if arr is None:
                raise AttributeError(
                    'Array should be specified in shared memory sender.')
            try:
                self.buf[:] = arr[:]
            except Exception:
                raise AttributeError(
                    'Size mismatch in shared memory receiver.')
        else:
            ret_arr = np.copy(
                np.ndarray(self.shape,
                           dtype=self.dtype,
                           buffer=self.shared_memory.buf))
            return ret_arr

    def close(self):
        self.shared_memory.close()
        self.shared_memory.unlink()


def to_dtype(s):
    if s == "bool":
        return bool
    else:
        return getattr(np, s)


class RGBDCameraBase(object):

    def __init__(self,
                 logger_name: str = "RGBD Camera",
                 shm_name_rgb: str = None,
                 shm_name_depth: str = None,
                 streaming_freq: int = 30,
                 **kwargs):
        '''
        Initialization.
        
        Parameters:
        - logger_name: str, optional, default: "RGBDCamera", the name of the logger;
        - shm_name_rgb: str, optional, default: None, the shared memory name of the camera RGB data, None means no shared memory object for RGB data;
        - shm_name_depth: str, optional, default: None, the shared memory name of the camera depth data, None means no shared memory object for depth data;
        - streaming_freq: int, optional, default: 30, the streaming frequency.
        '''
        super(RGBDCameraBase, self).__init__()
        #logging.setLoggerClass(ColoredLogger)
        self.logger = logging.getLogger(logger_name)
        self.is_streaming = False
        self.with_streaming_rgb = (shm_name_rgb is not None)
        self.with_streaming_depth = (shm_name_depth is not None)
        self.with_streaming = self.with_streaming_rgb or self.with_streaming_depth
        self.streaming_freq = streaming_freq
        self.shm_name_rgb = shm_name_rgb
        self.shm_name_depth = shm_name_depth
        self._prepare_shm()

    def _prepare_shm(self):
        '''
        Prepare shared memory objects.
        '''
        if self.with_streaming:
            rgb, depth = self.get_info()
            rgb = np.array(rgb).astype(np.uint8)
            depth = np.array(depth).astype(np.float32)
            if self.with_streaming_rgb:
                self.shm_camera_rgb = SharedMemoryManager(
                    self.shm_name_rgb, 0, rgb.shape, rgb.dtype)
                self.shm_camera_rgb.execute(rgb)
            if self.with_streaming_depth:
                self.shm_camera_depth = SharedMemoryManager(
                    self.shm_name_depth, 0, depth.shape, depth.dtype)
                self.shm_camera_depth.execute(depth)

    def streaming(self, delay_time=0.0):
        '''
        Start streaming.
        
        Parameters:
        - delay_time: float, optional, default: 0.0, the delay time before collecting data.
        '''
        if self.with_streaming is False:
            raise AttributeError(
                'If you want to use streaming function, either "shm_name_rgb" attribute or "shm_name_depth" attribute should be set correctly.'
            )
        self.thread = threading.Thread(target=self.streaming_thread,
                                       kwargs={'delay_time': delay_time})
        self.thread.setDaemon(True)
        self.thread.start()

    def streaming_thread(self, delay_time=0.0):
        time.sleep(delay_time)
        self.is_streaming = True
        self.logger.info('Start streaming ...')
        while self.is_streaming:
            rgb, depth = self.get_info()
            rgb = np.array(rgb).astype(np.uint8)
            depth = np.array(depth).astype(np.float32)
            if self.with_streaming_rgb:
                self.shm_camera_rgb.execute(rgb)
            if self.with_streaming_depth:
                self.shm_camera_depth.execute(depth)
            time.sleep(1.0 / self.streaming_freq)

    def stop_streaming(self, permanent=True):
        '''
        Stop streaming process.

        Parameters:
        - permanent: bool, optional, default: True, whether the streaming process is stopped permanently.
        '''
        self.is_streaming = False
        self.thread.join()
        self.logger.info('Close streaming.')
        if permanent:
            self._close_shm()
            self.with_streaming = False

    def _close_shm(self):
        '''
        Close shared memory objects.
        '''
        if self.with_streaming_rgb:
            self.shm_camera_rgb.close()
        if self.with_streaming_depth:
            self.shm_camera_depth.close()

    def get_info(self):
        '''
        Get the camera observation (RGB-D).
        '''
        return np.array([]), np.array([])


class RealSenseRGBDCamera(RGBDCameraBase):
    '''
    RealSense RGB-D Camera.
    '''

    def __init__(self,
                 serial,
                 frame_rate=30,
                 resolution=(1280, 720),
                 enable_emitter=True,
                 align=True,
                 logger_name: str = "RealSense RGBD Camera",
                 shm_name_rgb: str = None,
                 shm_name_depth: str = None,
                 streaming_freq: int = 30,
                 **kwargs):
        '''
        Initialization.

        Parameters:
        - serial: str, required, the serial number of the realsense device;
        - frame_rate: int, optional, default: 15, the framerate of the realsense camera;
        - resolution: (int, int), optional, default: (1280, 720), the resolution of the realsense camera;
        - enable_emitter: bool, optional, default: True, whether to enable the emitter;
        - align: bool, optional, default: True, whether align the frameset with the RGB image;
        - logger_name: str, optional, default: "Camera", the name of the logger;
        - shm_name: str, optional, default: None, the shared memory name of the camera data, None means no shared memory object;
        - streaming_freq: int, optional, default: 30, the streaming frequency.
        '''
        super(RealSenseRGBDCamera, self).__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.serial = serial
        # =============== Support L515 Camera ============== #
        self.is_radar = str.isalpha(serial[0])
        depth_resolution = (1024, 768) if self.is_radar else resolution
        if self.is_radar:
            frame_rate = max(frame_rate, 30)
            self.depth_scale = 4000
        else:
            self.depth_scale = 1000
        # ================================================== #
        # Set up device and stream
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.depth, depth_resolution[0],
                                  depth_resolution[1], rs.format.z16,
                                  frame_rate)
        self.config.enable_stream(rs.stream.color, resolution[0],
                                  resolution[1], rs.format.rgb8, frame_rate)
        # Start pipeline
        pipeline_profile = self.pipeline.start(self.config)
        # Set up emitter
        depth_sensor = pipeline_profile.get_device().query_sensors()[0]
        depth_sensor.set_option(rs.option.emitter_enabled, int(enable_emitter))
        # Set up alignment
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.with_align = align
        # Get intrinsic
        color_profile = pipeline_profile.get_stream(rs.stream.color)
        self.intrinsic = color_profile.as_video_stream_profile(
        ).get_intrinsics()
        super(RealSenseRGBDCamera,
              self).__init__(logger_name=logger_name,
                             shm_name_rgb=shm_name_rgb,
                             shm_name_depth=shm_name_depth,
                             streaming_freq=streaming_freq,
                             **kwargs)

    def get_rgb_image(self):
        '''
        Get the RGB image from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)
        return color_image

    def get_depth_image(self):
        '''
        Get the depth image from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data()).astype(
            np.float32) / self.depth_scale
        return depth_image

    def get_info(self):
        '''
        Get the RGB image along with the depth image from the camera.
        '''
        frameset = self.pipeline.wait_for_frames()
        if self.with_align:
            frameset = self.align.process(frameset)
        color_image = np.asanyarray(
            frameset.get_color_frame().get_data()).astype(np.uint8)
        depth_image = np.asanyarray(
            frameset.get_depth_frame().get_data()).astype(
                np.float32) / self.depth_scale
        return color_image, depth_image

    def get_intrinsic(self, return_mat=True):
        if return_mat:
            return np.array(
                [[self.intrinsic.fx, 0., self.intrinsic.ppx],
                 [0., self.intrinsic.fy, self.intrinsic.ppy], [0., 0., 1.]],
                dtype=np.float32)
        else:
            return self.intrinsic


camera_id = "image_mid"
camera_serial = "104122060902"
camera_cfg = {
    "frame_rate": 30,
    "resolution": [1280, 720],
    "enable_emitter": True,
    "align": True,
    "logger_name": "Camera-mid",
    "shm_name_rgb": "image_mid",
    "shm_name_depth": "depth_mid",
    "streaming_freq": 30,
}

camera = RealSenseRGBDCamera(serial=camera_serial, **camera_cfg)
for _ in range(30):
    camera.get_info()
print("Initialization Finished.")

ref_image_filepath = ""
ref_image = cv2.imread(ref_image_filepath)

while True:
    color, depth = camera.get_info()
    color_ = color[..., ::-1]
    alpha = 0.4
    mixed = cv2.addWeighted(color_, alpha, ref_image, 1 - alpha, 0)
    mixed = cv2.resize(mixed, (None, None), fx=0.8, fy=0.8)

    cv2.imshow("x", mixed)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    time.sleep(0.05)
