import cv2
import os
import time
import random
import json
import shutil
from collections import defaultdict
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

from yolo import YOLO
from preprocessing import handDetector, resize, skinDetector
from select_final import selectFinal


# --------------------------------------------------------------------------- #
# GPU CONFIGURATION
# --------------------------------------------------------------------------- #
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# --------------------------------------------------------------------------- #
# PREDICTION HELPER FUNCTIONS
# --------------------------------------------------------------------------- #
def decode_predictions(preds, top=5, model_json=""):
    CLASS_INDEX = json.load(open(model_json, "r", encoding="utf-8"))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
            each_result = [CLASS_INDEX[str(i)], pred[i]]
            results.append(each_result)
    return results


def preprocess_input(x):
    x *= (1.0 / 255.0)
    return x


def predictImage(image_input, jsonPath, result_count):
    image_to_predict = image.load_img(image_input, target_size=(224, 224))
    image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
    image_to_predict = np.expand_dims(image_to_predict, axis=0)
    image_to_predict = preprocess_input(image_to_predict)
    prediction = model.predict(x=image_to_predict, steps=1)
    
    prediction_results = []
    prediction_probabilities = []
    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=jsonPath)
    for result in predictiondata:
        prediction_results.append(str(result[0]))
        prediction_probabilities.append(str(result[1] * 100))
    return prediction_results, prediction_probabilities


# --------------------------------------------------------------------------- #
# STAGE 0: DETECT FACE FRAMES FOR ACTIVATION
# --------------------------------------------------------------------------- #
print("Stage 0: Detect Face Frames for Activation")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
color_blue = (255, 0, 0)
color_green = (0, 255, 0)

cap = cv2.VideoCapture(0)
cap.set(3, 864)
cap.set(4, 480)

face_detect = False
face_detected_count = 0

while not face_detect:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5)

        if len(faces) != 0:
            face_detected_count += 1
            text = "Face detected"
            cv2.putText(frame, text, (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_blue, 1)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_green, 1)
        else:
            if face_detected_count > 0:
                face_detected_count -= 1

        cv2.imshow("Face Detection", frame)
        cv2.waitKey(1)

        if face_detected_count >= 35:
            face_detect = True
            face_detected_count = 0
    else:
        if face_detected_count > 0:
            face_detected_count -= 1
        continue

cap.release()
cv2.destroyAllWindows()

video_folder = 'videos'
frames_folder = 'frames'
boxed_frames_folder = 'frames_boxed'
segmented_frames_folder = 'frames_segmented'

os.makedirs(video_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(boxed_frames_folder, exist_ok=True)
os.makedirs(segmented_frames_folder, exist_ok=True)


# --------------------------------------------------------------------------- #
# STAGE 1: RECORD WEBCAM FEED CONTAINING THE HAND SIGN
# --------------------------------------------------------------------------- #
print("Stage 1: Record Webcam Feed containing the Hand Sign")

clip_timestamp = str(time.time()).split('.')[0]
clip_path = os.path.join(video_folder, clip_timestamp + ".mp4")
video_codec = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24

cap1 = cv2.VideoCapture(0)
cap1.set(3, 864)
cap1.set(4, 480)

video_writer = cv2.VideoWriter(clip_path, video_codec, fps, (int(cap1.get(3)), int(cap1.get(4))))
start_time = time.time()
done = 0

while cap1.isOpened():
    ret, frame = cap1.read()
    if ret:
        if done == 0:
            frame = cv2.flip(frame, 1)
            cv2.imshow("Webcam Feed", frame)
            cv2.waitKey(1)

            if time.time() - start_time >= 7:
                clip_timestamp2 = str(time.time()).split('.')[0]
                clip_path = os.path.join(video_folder, clip_timestamp2 + ".mp4")
                video_writer = cv2.VideoWriter(clip_path, video_codec, fps, (int(cap1.get(3)), int(cap1.get(4))))
                done = 1

            video_writer.write(frame)

        if done == 1:
            break

cap1.release()
cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
# STAGE 2: EXTRACT FRAMES FROM THE CLIP
# --------------------------------------------------------------------------- #
print("Stage 2: Extract Frames from the Clip")

clip_loc = os.path.join(video_folder, clip_timestamp + ".mp4")
cap2 = cv2.VideoCapture(clip_loc)
num_of_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
num_of_frames_to_select = max(1, int(num_of_frames / 5))

frames_to_select = []
frame_counter = 0
while frame_counter < num_of_frames_to_select and num_of_frames > 0:
    r = random.randint(1, num_of_frames)
    if r not in frames_to_select:
        frames_to_select.append(r)
        frame_counter += 1

frame_index = 1
frame_counter_saved = 0
while frame_counter_saved < frame_counter:
    ret, frame = cap2.read()
    if ret:
        if frame_index in frames_to_select:
            frame_counter_saved += 1
            frame_name = clip_timestamp + '_' + str(frame_counter_saved) + '.jpg'
            frame_loc = os.path.join(frames_folder, frame_name)
            cv2.imwrite(frame_loc, frame)
        frame_index += 1
    else:
        break

cap2.release()
cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
# STAGE 3: PREPROCESS THE FRAMES
# --------------------------------------------------------------------------- #
print("Stage 3: Preprocess the Frames")

yolo = YOLO("yolo_models/cross-hands.cfg", "yolo_models/cross-hands.weights", ["hand"])
yolo.confidence = 0.5

_, _, files = next(os.walk(frames_folder))

for f in files:
    filename = str(f)
    if filename.split('_')[0] == clip_timestamp:
        input_file_location = os.path.join(frames_folder, filename)
        output_file_location = os.path.join(boxed_frames_folder, filename)

        frame = cv2.imread(input_file_location)
        if frame is None:
            continue

        x, y, x1, y1 = handDetector.hand_seg(input_file_location, output_file_location, yolo, color_green)

        if x is not None:
            output_file_location = os.path.join(segmented_frames_folder, filename)

            x = max(0, x - 54)
            y = max(0, y - 30)
            x1 = min(frame.shape[1], x1 + 54)
            y1 = min(frame.shape[0], y1 + 30)

            cropped_frame = frame[y:y1, x:x1]
            if cropped_frame.size == 0:
                continue

            resized_frame = resize.resize_image(cropped_frame, (224, 224))
            detector = skinDetector(resized_frame)
            output_frame = detector.find_skin()
            cv2.imwrite(output_file_location, output_frame)


# --------------------------------------------------------------------------- #
# STAGE 4: MODEL LOADING & INFERENCE (UNIVERSAL KERAS 3 SUPPORT)
# --------------------------------------------------------------------------- #
print("Stage 4: Use the Trained Model to Detect the Sign Displayed")

def resolve_model_path():
    """Finds the model file or directory under App folder."""
    candidates = [
        "final_model.h5",
        "final_model.keras",
        "final_model",
        os.path.join("App", "final_model.h5"),
        os.path.join("App", "final_model.keras"),
        os.path.join("App", "final_model")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Could not locate 'final_model', 'final_model.h5', or 'final_model.keras'.\n"
        "Please ensure the model file is placed in your project root or App folder."
    )

def load_robust_model():
    target = resolve_model_path()
    print(f"Loading model from: {target}")

    # Case 1: SavedModel folder
    if os.path.isdir(target):
        try:
            return keras.layers.TFSMLayer(target, call_endpoint="serving_default")
        except Exception:
            saved = tf.saved_model.load(target)
            endpoint = list(saved.signatures.keys())[0]
            del saved
            return keras.layers.TFSMLayer(target, call_endpoint=endpoint)

    # Case 2: Standalone file (.h5, .keras, or extension-less H5 file)
    if os.path.isfile(target):
        # If it's extension-less, rename temporarily or load directly via h5
        if not target.endswith((".h5", ".keras")):
            renamed_target = target + ".h5"
            if not os.path.exists(renamed_target):
                shutil.copyfile(target, renamed_target)
            target = renamed_target
        
        return keras.models.load_model(target, compile=False)

class UniversalPredictor:
    def __init__(self):
        self.model_obj = load_robust_model()

    def predict(self, x, steps=1):
        if hasattr(self.model_obj, "predict"):
            return self.model_obj.predict(x, steps=steps)
        
        # TFSMLayer inference fallback
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        out = self.model_obj(x_tensor)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        if hasattr(out, "numpy"):
            out = out.numpy()
        return out

model = UniversalPredictor()

_, _, files = next(os.walk(segmented_frames_folder))
opcount = defaultdict(int)

for f in files:
    filename = str(f)
    if filename.split('_')[0] == clip_timestamp:
        input_file_location = os.path.join(segmented_frames_folder, filename)
        opcount[predictImage(input_file_location, 'model_class.json', 1)[0][0]] += 1

print("\nFrequency Count of Detections:", dict(opcount))

if opcount:
    detected_sign = selectFinal.select_max(opcount)
    print("Output: ", detected_sign)
else:
    print("No hand signs were detected in the segmented frames.")