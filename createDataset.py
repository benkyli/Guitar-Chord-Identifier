import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # removed a warning about floating point problems
import itertools
import numpy as np
import csv
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)           # assumes your webcam is webcam 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()             # default method captures 2 hands; 1 hand may be preferable 
mpDraw = mp.solutions.drawing_utils # method for showing hand joints on camera

# get dataset csv path
root = os.path.dirname(__file__)
dataset_path = os.path.join(root, 'chordDataset.csv')


def main():
    # Get user input for what chord they are creating samples for
    chord = input('Enter the chord you want to sample: ')
    
    # collect chord sample data
    max_frames = 400
    for frame in range(max_frames):
        # read feed
        ret, img = cap.read()
        if not ret:     # ensures that webcam is working
            break
        
        # show preparation message as program starts
        if frame == 0:
            # parameters:(image, text, position, font, font size, colour, line thickness, line type)
            cv2.putText(img, "STARTING COLLECTION", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128,0,128), 1, cv2.LINE_AA)
            cv2.imshow("Image", img)
            cv2.waitKey(5000)   # wait 5 seconds to give user time to prepare
        
        # show number of samples collected
        else:
            cv2.putText(img, f"Samples collected: {frame}", (100,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        # get image data
        results = process_hands(img, hands)           
        if results.multi_hand_landmarks:
            for handLMs, handedness in zip(results.multi_hand_landmarks,
                            results.multi_handedness):
                # draw hand landmarks on image
                mpDraw.draw_landmarks(img, handLMs, mpHands.HAND_CONNECTIONS) 

                # get landmark point data
                lm_points = calc_landmark_points(img, handLMs)

                # write to dataset file
                hand_class = handedness.classification[0].index
                csv_append(chord, hand_class, lm_points)

        cv2.imshow("Image", img)

        # Press ESC to close program
        if cv2.waitKey(1) == 27:
            break

    # show completion message
    ret, img = cap.read()
    cv2.putText(img, "Sampling complete", (100, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128,0,128), 1, cv2.LINE_AA)
    cv2.imshow("Image", img)
    cv2.waitKey(3000)

    cap.release()
    cv2.destroyAllWindows()


def process_hands(img, hands):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # convert colour from BGR to RGB
    imgRGB.flags.writeable = False                  # make image unwriteable to save memory
    results = hands.process(imgRGB)                 # make predictions for hand
    return results


def calc_landmark_points(img, landmarks):
    # turn landmarks into keypoints
    img_width, img_height = img.shape[1], img.shape[0]
    landmark_points = []
    for landmark in landmarks.landmark:
        lm_x = min(int(landmark.x * img_width), img_width - 1)
        lm_y = min(int(landmark.y * img_height), img_height - 1)
        landmark_points.append([lm_x, lm_y])

    # convert keypoints into relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(landmark_points):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        landmark_points[index][0] = landmark_points[index][0] - base_x
        landmark_points[index][1] = landmark_points[index][1] - base_y

    # convert to one-dimensional numpy array
    landmark_points = np.array(list(
        itertools.chain.from_iterable(landmark_points)))
    
    # normalize points
    max_value = np.max(np.absolute(landmark_points))
    landmark_points = landmark_points / max_value
 
    return landmark_points


def csv_append(label, handedness, landmarks):
    with open(dataset_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, handedness, *landmarks])
    return


if __name__ == '__main__':
    main()