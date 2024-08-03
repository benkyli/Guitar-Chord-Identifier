import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # removed a warning about floating point problems
import json
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from createDataset import process_hands, calc_landmark_points

cap = cv2.VideoCapture(0)           # assumes your webcam is webcam 0

# setup hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()            
mpDraw = mp.solutions.drawing_utils 

# load model
model_name = 'EDGFA' # NOTE: You must input the exact model you want to use here. Do not include the file extension; the line below assumes that it is a keras model.
model = tf.keras.models.load_model(f'models/{model_name}.keras') 

# get decoding dictionary
decode_dict_path = f'Decoding JSON/{model_name}.json'
with open(decode_dict_path, 'r') as f:
    decode_dict = json.load(f)


def main():
    # collect live data
    while True:
        # read feed
        ret, img = cap.read()
        if not ret:     # ensures that webcam is working
            break

        # get image data
        results = process_hands(img, hands)           
        if results.multi_hand_landmarks:
            for handLMs, handedness in zip(results.multi_hand_landmarks,
                            results.multi_handedness):
                # draw hand landmarks on image
                mpDraw.draw_landmarks(img, handLMs, mpHands.HAND_CONNECTIONS) 

                # get landmark and hand data
                lm_points = calc_landmark_points(img, handLMs)
                hand_class = handedness.classification[0].index

                # use data to make a prediction on the chord being played
                data = np.array([hand_class, *lm_points])
                data.shape = (1, 43)
                prediction = model.predict(data)
                pred_index = np.argmax(prediction[0])
                predicted_chord = decode_dict[str(pred_index)]

                # display the chord prediction
                cv2.putText(img, f"Prediction: {predicted_chord}", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128,0,128), 1, cv2.LINE_AA)

        cv2.imshow("Image", img)

        # Press ESC to close program
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()