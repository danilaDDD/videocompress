from os.path import join
import time
from collections import deque

import cv2
import numpy as np


class Video(cv2.VideoCapture):
    codec = cv2.VideoWriter_fourcc(*'XVID')

    @staticmethod
    def __is_pressed_key(key):
        return cv2.waitKey(1) & 0xFF == ord(key)

    def __init__(self, file, skip=1, fps=50, frames=None, codec=None):
        if codec is not None:
            self.codec = codec

        if frames is not None:
            first_frame = np.uint8(next(frames))
            imshape = first_frame.shape
            a, b, _ = imshape
            if a != b:
                raise IOError('width != height frame!')

            out = cv2.VideoWriter(file, Video.codec, fps, imshape[:2])
            out.write(first_frame)

            for frame in frames:
                out.write(np.uint8(frame))

            out.release()

        super().__init__(file)
        self.fps = fps
        self.file = file
        self.skip = skip
        frame = self.get_frame(0)
        if frame is not None:
            self.__imshape = frame.shape
        else:
            raise IOError('frames[0]==None!')

    @property
    def imshape(self):
        return self.__imshape

    @property
    def fps(self):
        return self.__fps

    @fps.setter
    def fps(self, value):
        self.set(cv2.CAP_PROP_FPS, value)

    def track(self, skip=None):
        if skip is not None:
            self.skip = skip

        count = 0
        while self.isOpened():
            self.set(cv2.CAP_PROP_POS_FRAMES, count * self.skip)
            grab, frame = self.read()
            if grab:
                yield frame
            else:
                break

            if self.__is_pressed_key('q'):
                break

            count += 1

        else:
            raise IOError('not open!')
        self.release()

    def show(self, delay=0.1):
        print('show')
        count = 0
        if self.isOpened():
            for frame in self.track():
                cv2.imshow('frame', frame)
                time.sleep(delay)
                if self.__is_pressed_key('q'):
                    break
                print(count)
                count += 1
        else:
            raise IOError('not open')
        self.release()
        cv2.destroyAllWindows()

    def get_frame(self, idx):
        if self.isOpened():
            self.set(cv2.CAP_PROP_POS_FRAMES, idx * self.skip)
            grab, frame = self.read()
            if grab:
                return frame
            else:
                return None
        else:
            raise IOError('not open')

    def reopen(self, file=None):
        if file is None:
            file = self.file
        else:
            self.file = file

        self.open(file)
        return self.isOpened()

    def to_array(self):
        try:
            idx = 0
            frames = deque()
            while True:
                frame = self.get_frame(idx)

                if self.__is_pressed_key('q'):
                    break

                if frame is None:
                    break

                frames.append(frame)
                idx += 1

            return np.array(frames, dtype=np.uint8)
        except IOError as e:
            raise e

    def __str__(self):
        a, b, c = self.imshape
        return "f{a};{b};{c}"


