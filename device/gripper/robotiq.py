'''
Robotiq Gripper Interface.

Author: Anonymous, Hongjie Fang.

Refererences:
  [1] https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf
'''

import time
import struct
import serial
import threading
import numpy as np


class Robotiq2FGripper:
    '''
    Robotiq Gripper (2F-85, 2F-140) Interface
    '''

    def __init__(self, port: str = '/dev/ttyUSB0', **kwargs):
        '''
        Initialization.
        
        Parameters:
        - port: str, required, the port of the Robotiq 2F-(85/140) Gripper;
        - logger_name: str, optional, default: "Robotiq Gripper", the name of the logger;
        - shm_name: str, optional, default: None, the shared memory name of the gripper data, None means no shared memory object;
        - streaming_freq: int, optional, default: 30, the streaming frequency.
        '''
        super(Robotiq2FGripper, self).__init__()
        self.port = port
        self.waiting_gap = 0.01
        self.last_position = 255
        self.last_speed = 100
        self.last_force = 77
        self.last_timestamp = int(time.time() * 1000)
        self.ser = serial.Serial(port=self.port,
                                 baudrate=115200,
                                 timeout=1,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 bytesize=serial.EIGHTBITS)
        self.lock = threading.Lock()
        self.activate()
        self.close_gripper()
        self.open_gripper()
        print('Activated.')

    def activate(self):
        '''
        Activate the gripper.
        Refer to: page 62 of ref [1].
        '''
        # Activation Request
        self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30")
        response = self.ser.read(8)
        if response != b"\x09\x10\x03\xE8\x00\x03\x01\x30":
            raise AssertionError('Unexpected response of the gripper.')
        time.sleep(self.waiting_gap)
        self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x01\x00\x00\x00\x00\x00\x72\xE1")
        response = self.ser.read(8)
        if response != b"\x09\x10\x03\xE8\x00\x03\x01\x30":
            raise AssertionError('Unexpected response of the gripper.')
        time.sleep(self.waiting_gap)
        # Read Gripper status until the activation is completed
        self.ser.write(b"\x09\x03\x07\xD0\x00\x01\x85\xCF")
        while self.ser.read(7) != b"\x09\x03\x02\x31\x00\x4C\x15":
            time.sleep(self.waiting_gap)
            self.ser.write(b"\x09\x03\x07\xD0\x00\x01\x85\xCF")

    def open_gripper(self):
        '''
        Open the gripper at full speed and full force.
        Refer to: page 68 of ref [1].
        '''
        self.lock.acquire()
        self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19")
        response = self.ser.read(8)
        self.lock.release()
        if response != b"\x09\x10\x03\xE8\x00\x03\x01\x30":
            raise AssertionError('Unexpected response of the gripper.')
        time.sleep(self.waiting_gap)
        while self.get_info()[2] != 0:
            time.sleep(self.waiting_gap)

    def close_gripper(self):
        '''
        Close the gripper at full speed and full force.
        Refer to: page 65 of ref [1].
        '''
        self.lock.acquire()
        self.ser.write(b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\x64\x03\x82")
        response = self.ser.read(8)
        if response != b"\x09\x10\x03\xE8\x00\x03\x01\x30":
            raise AssertionError('Unexpected response of the gripper.')
        self.lock.release()
        time.sleep(self.waiting_gap)
        while self.get_info()[2] != 1:
            time.sleep(self.waiting_gap)

    def action(self, position, speed=100, force=77, **kwargs):
        '''
        Send the control command to the gripper.

        Parameters:
        - position: int, required, the position of the gripper (from 0 to 255);
        - speed: int, optional, default: 100, the speed of the gripper (from 0 to 255);
        - force: int, optional, default: 77 (30% of the force), the force of the gripper (from 0 to 255).
        '''
        position = int(min(255, max(0, position)))
        speed = int(min(255, max(0, speed)))
        force = int(min(255, max(0, force)))
        command = bytearray(b"\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\x00\x00")
        command[10] = position
        command[11] = speed
        command[12] = force
        self.lock.acquire()
        self.send_command(command)
        self.ser.read(8)
        self.lock.release()
        self.last_position = position
        self.last_force = force
        self.last_speed = speed
        self.last_timestamp = int(time.time() * 1000)

    def send_command(self, command):
        crc = self._calc_crc(command)
        data = command + crc
        self.ser.write((data))

    def get_info(self):
        '''
        Get the current information about the gripper (position, force, status, last command).
        Refer to: page 66-67, 69-70 of ref [1].
        
        Returns:
        - position: the position of the gripper, from 0 to 255.
        - force: the force of the gripper, from 0 to 255.
        - status:
          * 1: error raised;
          * 0: opening, completed;
          * 1: closing, completed;
          * 2: opening, not completed;
          * 3: closing, not completed;
        - position, force, speed, timestamp of the last command.
        '''
        while True:
            self.lock.acquire()
            self.ser.write(b"\x09\x03\x07\xD0\x00\x03\x04\x0E")
            data = self.ser.read(11)
            self.lock.release()
            # Not a valid response
            if data[:3] != b"\x09\x03\x06":
                time.sleep(self.waiting_gap)
                continue
            # Check error flag. Only allow "no fault" and "minor fault: no communication".
            if data[5] != 0x00 and data[5] != 0x09:
                return np.array(
                    [data[7], data[8], -1, self.last_position, self.last_force, self.last_speed,
                     self.last_timestamp]).astype(np.int64)
            # Complete Flag.
            if data[3] == 0xF9 or data[3] == 0xB9 or data[3] == 0x79:
                completed = True
            elif data[3] == 0x39:
                completed = False
            else:
                time.sleep(self.waiting_gap)
                continue
            # Open/close Flag.
            if data[6] == 0xFF:
                g_status = True
            else:
                g_status = False
            status = (1 - int(completed)) * 2 + (int(g_status))
            return np.array(
                [data[7], data[8], status, self.last_position, self.last_force, self.last_speed,
                 self.last_timestamp]).astype(np.int64)

    def _calc_crc(self, command):
        '''
        Calculate the Cyclic Redundancy Check (CRC) bytes for command.

        Parameters:
        - command: bytes, required, the given command.
        
        Returns:
        - The calculated CRC bytes.
        '''
        crc_registor = 0xFFFF
        for data_byte in command:
            tmp = crc_registor ^ data_byte
            for _ in range(8):
                if (tmp & 1 == 1):
                    tmp = tmp >> 1
                    tmp = 0xA001 ^ tmp
                else:
                    tmp = tmp >> 1
            crc_registor = tmp
        crc = bytearray(struct.pack('<H', crc_registor))
        return crc
