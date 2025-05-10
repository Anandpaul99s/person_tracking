from deepface import DeepFace
import cv2
import numpy as np


class FaceEmbedder:
    def __init__(self):
        self.model_name = 'Facenet'  # Options: Facenet, VGG-Face, DeepFace, OpenFace

    def get_embedding(self, face_image):
        """
        Extracts embedding from a cropped face image.
        """
        try:
            embedding = DeepFace.represent(
                face_image, model_name=self.model_name, enforce_detection=False)
            if embedding and 'embedding' in embedding[0]:
                return np.array(embedding[0]['embedding'])
        except Exception as e:
            print(f"[Embedding Error] {e}")
        return None
