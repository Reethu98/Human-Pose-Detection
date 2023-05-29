import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import datetime


CONFIDENCE_THRESHOLD_WEB_CAM = 0.5
CONFIDENCE_THRESHOLD_IMAGE = 0.3
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
# model_path = "/desktop/review/movenet_multipose_lightning_1"
# model = tf.saved_model.load(model_path)
print("close")
movenet = model.signatures['serving_default']


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)
            cv2.putText(frame, text=str(kp_conf*100)[:2], org=(int(kx), int(ky)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.4, color=(0, 0, 0), thickness=1)

con1, con2, count = 0, 0, 0
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    c1_sum, c2_sum = 0, 0
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
         
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):  
            c1_sum += c1
            c2_sum += c2
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

    PDJ = round((c1_sum+c2_sum)*100/30, 2) #Percentage of Detected Joints
    if c1_sum:
        print(f'Percentage of Detected Joints - {PDJ}%')

def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

def draw_on_image():
    IMAGE_FILES = ["004.jpeg", "1.jpeg", "2.jpeg","009.jpg","014.jpg","020.jpg","026.jpg","002.jpg","003.jpg","004.jpg","005.jpg","025.jpg","037.jpg","038.jpg","039.jpg","040.jpg"]
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(f"images/{file}")
        if type(image)==type(None):
            continue
        frame = image.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 384,640)
        input_img = tf.cast(img, dtype=tf.int32)
        
        
        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        
        loop_through_people(frame, keypoints_with_scores, EDGES, CONFIDENCE_THRESHOLD_IMAGE)
        
        cv2.imshow('Movenet Multipose', frame)
        cv2.imwrite(f"output/{file}", frame)
        
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        time.sleep(1)

def plot_on_web_cam(cap):
    ret, frame = cap.read()
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
    input_img = tf.cast(img, dtype=tf.int32)
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))

    loop_through_people(frame, keypoints_with_scores, EDGES, CONFIDENCE_THRESHOLD_WEB_CAM)
    cv2.imshow('Movenet Multipose', frame)
    

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
detection, detection_stopped_time, timer_started, seconds_to_record_after_detection  = False, None, False, 5

draw_on_image()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    plot_on_web_cam(cap)
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in bodies:
        cv2.imshow("webcam", frame)

    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(
                f"./video/{current_time}.mp4", fourcc, 20, frame_size)
            print("Started Recording")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= seconds_to_record_after_detection:
                detection = False
                timer_started = False
                out.release()
                print('Stop Recording')
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


