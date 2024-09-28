import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import time
import pyttsx3
import pickle
import cvlib as cv
from gtts import gTTS
from googletrans import Translator
import io
import pygame
engine = pyttsx3.init()
from keras.preprocessing.image import img_to_array
# res = [0.7, 0.2, 0.1]

from tensorflow.keras.models import load_model
language = 'en'

model_path = "actionf.h5"
model = load_model(model_path)

model_dict = pickle.load(open('./model.p', 'rb'))
model2 = model_dict['model']

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

model3 = load_model('gender.h5')


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)               
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 

def draw_styled_landmarks(image, results):
    # face
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # hand left
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    #  right 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# colors = [(245,117,16), (117,245,16), (16,117,245)]
# def prob_viz(res, actions, input_frame, colors):
#     output_frame = input_frame.copy()
#     for num, prob in enumerate(res):
#         cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
#         cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
#     return output_frame


engine = pyttsx3.init()
pygame.mixer.init()

def speak_text(text, language, gender):
    global engine  
    gender = genders
    if gender == 'male':
        voices = engine.getProperty('voices')
        male_voice = None
        for voice in voices:
            if 'male' in voice.name.lower():
                male_voice = voice
                break
        if male_voice is None:
            male_voice = voices[0]  
        
        engine.setProperty('voice', male_voice.id)
        
        engine.say(text)
        engine.runAndWait()
        
    else:
        translator = Translator()
        
        translated = translator.translate(text, dest=language)
        translated_text = translated.text

        tts = gTTS(text=translated_text, lang=language, slow=False)
        
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


genders = None

def gender():
    webcam = cv2.VideoCapture(0)
    classes = ['man', 'woman']
    threshold = 95
    genders = None  

    while webcam.isOpened():
        status, frame = webcam.read()

        face, confidence = cv.detect_face(frame)

        if len(face) > 0:
            for idx, f in enumerate(face):
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                face_crop = np.copy(frame[startY:endY, startX:endX])

                if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                    continue

                face_crop = cv2.resize(face_crop, (96, 96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                conf = model3.predict(face_crop)[0]
                idx = np.argmax(conf)
                label = classes[idx]
                label = "{}: {:.2f}%".format(label, conf[idx] * 100)

                if conf[idx] * 100 > threshold:
                    genders = classes[idx]
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    break

        cv2.imshow("gender detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if genders is not None:
            print(f"Detected gender: {genders}")
            break
    webcam.release()
    cv2.destroyAllWindows()

    if genders is not None:
        text = f"Detected Gender Type is {genders}"
        speak_text(text, language, genders)

gender()




def action_model():
    action_text = "Switching to American Sign Language Model"
    DATA_PATH = os.path.join('MP_Data') 
    actions = np.array(['hello', 'welcome', 'to', 'our', 'project', 'indian', 'sign', 'language'])  
    no_sequences = 80
    sequence_length = 30

    sequence = []
    sentence = []
    threshold = 0.70

    current_action_index = 0  # To track the current expected action

    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)

            draw_styled_landmarks(image, results)

            # Check if face and right hand are detected
            if results.right_hand_landmarks and results.face_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predicted_action = actions[np.argmax(res)]

                    # Give preference to the next action in the sequence
                    if predicted_action == actions[current_action_index] and res[np.argmax(res)] > threshold:
                        # Add the correct prediction to the sentence
                        if len(sentence) == 0 or predicted_action != sentence[-1]:
                            sentence.append(predicted_action)
                            speak_text(predicted_action, language, genders)

                            # Move to the next expected action in the sequence
                            current_action_index += 1
                            if current_action_index >= len(actions):
                                current_action_index = 0  # Reset to the beginning of the sequence if completed

            # Check if both hands are detected (can apply this logic similarly if both hands are needed for certain actions)
            if results.right_hand_landmarks and results.left_hand_landmarks and results.face_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predicted_action = actions[np.argmax(res)]

                    # Give preference to the next action in the sequence
                    if predicted_action == actions[current_action_index] and res[np.argmax(res)] > threshold:
                        # Add the correct prediction to the sentence
                        if len(sentence) == 0 or predicted_action != sentence[-1]:
                            sentence.append(predicted_action)
                            speak_text(predicted_action, language, genders)

                            # Move to the next expected action in the sequence
                            current_action_index += 1
                            if current_action_index >= len(actions):
                                current_action_index = 0  # Reset to the beginning of the sequence if completed

            # Limit the length of the sentence to the last 5 actions
            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Display the live feed with predictions
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def sign_model():
    sign_text = "Switching to Indian Sign Language Model"
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing2 = mp.solutions.drawing_utils
    mp_drawing2_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

    labels_dict = {0: 'A', 1: 'B', 2: 'L'}
    tracked_hand_wrist = None

    current_letter = None
    letter_start_time = None
    collected_letters = []
    pause_start_time = None
    displayed_word = "" 

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y

            if tracked_hand_wrist is None or abs(tracked_hand_wrist[0] - wrist_x) > 0.1 or abs(tracked_hand_wrist[1] - wrist_y) > 0.1:
                tracked_hand_wrist = (wrist_x, wrist_y)

            if (tracked_hand_wrist[0] - wrist_x) < 0.05 and (tracked_hand_wrist[1] - wrist_y) < 0.05:
                mp_drawing2.draw_landmarks(
                    frame,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing2_styles.get_default_hand_landmarks_style(),
                    mp_drawing2_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model2.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                if current_letter != predicted_character:
                    current_letter = predicted_character
                    letter_start_time = time.time()
                else:
                    if time.time() - letter_start_time >= 1:
                        collected_letters.append(current_letter)
                        letter_start_time = time.time()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

                pause_start_time = None
        else:
            tracked_hand_wrist = None
            
            if pause_start_time is None:
                pause_start_time = time.time()
            elif time.time() - pause_start_time > 1:
                if collected_letters:
                    displayed_word = ''.join(collected_letters)
                    print("Collected Word:", displayed_word)

                    cv2.putText(frame, displayed_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
                    
                    speak_text(displayed_word, language, genders)  

                    if displayed_word == "A":
                        cap.release()
                        cv2.destroyAllWindows()
                        speak_text(sign_text, language, genders) 
                        action_model()

                    collected_letters = []  
        if displayed_word:
            cv2.putText(frame, displayed_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
action_model()
# sign_model()

# Hindi - hi
# Bengali - bn
# Tamil - ta
# Telugu - te
# Kannada - kn
# Gujarati - gu
# Marathi - mr
# Punjabi - pa
# Malayalam - ml
# Odia - or
# Maithili (mai)
# Bhojpuri (bho) 
# Magahi (mag)

