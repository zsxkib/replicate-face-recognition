# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import face_recognition
import dlib
from typing import List
import cv2
import logging
from cog import BasePredictor, Input, Path

logging.basicConfig(level=logging.INFO)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.use_gpu = dlib.cuda.get_num_devices() > 0
        if self.use_gpu:
            logging.info("dlib is using GPU", dlib.DLIB_USE_CUDA)
        else:
            logging.info("dlib is NOT using GPU", dlib.DLIB_USE_CUDA)

    def process_batch(self, frames):
        batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)
        output = []

        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
            number_of_faces_in_frame = len(face_locations)

            frame_number = len(frames) - 128 + frame_number_in_batch

            for face_location in face_locations:
                top, right, bottom, left = face_location
                logging.debug(
                    " - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
                        top, left, bottom, right
                    )
                )

                face_image = frames[frame_number][top:bottom, left:right]
                face_image_path = f"face_{frame_number}_{face_location}.jpg"
                cv2.imwrite(face_image_path, face_image)
                output.append(Path(face_image_path))

        return output

    def predict(self, input_file: Path = Input(description="Input video or image file")) -> List[Path]:
        """Run a single prediction on the model"""
        input_file_capture = cv2.VideoCapture(str(input_file))

        frames = []
        output = []

        while input_file_capture.isOpened():
            ret, frame = input_file_capture.read()

            if not ret:
                break

            frame = frame[:, :, ::-1]
            frames.append(frame)

            if len(frames) == 128:
                output.extend(self.process_batch(frames))
                frames = []

        input_file_capture.release()
        return output
