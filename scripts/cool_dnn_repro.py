"""Minimal reproduction: OpenCV DNN text recognition (OpenCV Zoo CRNN) on one 100x32 crop.

Usage:  python -X faulthandler scripts/cool_dnn_repro.py
Prints the OpenCV build, the model's SHA-256 and the recognised text; with COOL 3.1 on Graviton4
this aborted with a heap error ("malloc(): invalid size" / "double free or corruption").
"""
import hashlib
import os
import platform

import cv2 as cv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = os.path.join(ROOT, "models", "crnn_en.onnx")

print("cv2", cv.__version__, cv.__file__, "| python", platform.python_version(), "|", platform.machine(), flush=True)
print("model sha256", hashlib.sha256(open(MODEL, "rb").read()).hexdigest(), flush=True)
img = np.full((32, 100, 3), 255, np.uint8)
cv.putText(img, "20", (30, 26), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
print("step 1: load model", flush=True)
rec = cv.dnn.TextRecognitionModel(MODEL)
print("step 2: configure", flush=True)
rec.setDecodeType("CTC-greedy")
rec.setVocabulary(list("0123456789abcdefghijklmnopqrstuvwxyz"))
rec.setInputParams(1 / 127.5, (100, 32), (127.5, 127.5, 127.5))
print("step 3: recognize", flush=True)
print("text:", repr(rec.recognize(img)), flush=True)
print("step 4: done", flush=True)
