from __future__ import unicode_literals
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import cv2

mp_hands = mp.solutions.hands # Hands model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection_hands(image, model):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # for mobile camera
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # for web camera
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw left hand connections
    if results.multi_hand_landmarks and results.multi_handedness:
        for index in range(len(results.multi_hand_landmarks)) :
            classification = results.multi_handedness[index].classification
            if classification[0].label == 'Right':
                mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[index], mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        ) 
            # Draw right hand connections  
            else :
                mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[index], mp_hands.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    ) 
# get left hand and right hand landmarks if there.
def extract_keypoints(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    for index in range(len(results.multi_hand_landmarks)) :
        classification = results.multi_handedness[index].classification
        if classification[0].label == 'Right':
            rh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[index].landmark]).flatten()
        else :
            lh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[index].landmark]).flatten()    
    return np.concatenate([lh, rh])

def prob_viz(res, action, input_frame):
    output_frame = input_frame.copy()
    cv2.rectangle(output_frame, (0,60), (int(res[1]*100), 90), (245,117,16), -1)
    cv2.putText(output_frame, action, (0, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

def main():

    modelname = input("model name : " ) # name of the model
    # Load actions array from CSV file
    actions = np.genfromtxt(modelname+'.txt',dtype=None, delimiter=',',encoding='UTF-8') 
    print(actions)
    model = load_model(modelname) #loading model
    no_frames = int(input("number of frames per sequence for prediction: " ))
    # Cam source that you use (normally 0)
    no_cam = int(input("you cam source number (try 0 or 1 or 2): " ))
    threshold = float(input("accuracy threshold: " ))
    stability_coff = int(input("get highest prediction in last ... : "))

    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []

    cap = cv2.VideoCapture(no_cam)
    # Set mediapipe model 
    with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.7,min_tracking_confidence=0.5) as hands :
        while cap.isOpened():

            # Read feed
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Make detections
            image, results = mediapipe_detection_hands(frame, hands)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            if results.multi_hand_landmarks and results.multi_handedness:
                keypoints = extract_keypoints(results)
                # ignore frames with no hands
                if not np.array_equal(keypoints , np.zeros(126)):
                    sequence.append(keypoints)
                    # sequence = sequence[-30:]
            
                # do predictions when enough frames are aquired
                if len(sequence) == no_frames:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    if np.amax(res) > threshold: 
                        predictions.append(np.argmax(res))
                    
                    sequence = [] #empty sequence to collect new frames
                

                    #3 Viz logic
                    if len(predictions)>= int(stability_coff) and np.unique(predictions[-stability_coff:])[0]==np.argmax(res): 
                        predictions = predictions[int(-stability_coff):]
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                print(sentence[-1])
                        else:
                            sentence.append(actions[np.argmax(res)])
                            print(sentence[-1])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]
            # Viz probabilities
            if len(sentence) > 0:
                image = prob_viz((np.argmax(res),np.amax(res)), actions[np.argmax(res)], image)

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
