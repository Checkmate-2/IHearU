from __future__ import unicode_literals
import cv2
import numpy as np
import os
import mediapipe as mp

mp_hands = mp.solutions.hands # Hands model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection_hands(image, model):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # for web
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # for mobile
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks and results.multi_handedness:
        for index in range(len(results.multi_hand_landmarks)) :
            classification = results.multi_handedness[index].classification
            # Draw right hand connections  
            if classification[0].label == 'Right':
                mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[index], mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        ) 
            # Draw left hand connections
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

def main():

    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join(input("data folder : " )) 

    # Action that we try to detect
    action = input("action name : " )

    # number of videos 
    no_sequences = int(input("number of sequences : " ))

    # Videos are going to be .. frames in length
    no_frames = int(input("number of frames per sequence : " ))

    # Cam source that you use (normally 0)
    no_cam = int(input("you cam source number (try 0 or 1 or 2): " ))

    #Creat folder for the action if there's not
    try:
        os.makedirs(os.path.join(DATA_PATH, action))
    except:
        pass

    # data collection  
    # capturing video source  
    cap = cv2.VideoCapture(no_cam)
    # Set mediapipe model 
    with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.7,min_tracking_confidence=0.5) as hands :
        
        # Loop through actions
        cv2.waitKey(2000) #at the start wait 2 seconds
        window = [] # the list that will contain all the landmarks
        # Loop through sequences aka videos
        for sequence in range(no_sequences):   
            frame_list = []
            frame_num = 0
            # Loop through video length aka number of frames
            while frame_num < no_frames: 
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

                # Apply wait logic
                if sequence == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Video Sequence Number {}'.format(sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                # tracking the number of vides being captured    
                else: 
                    cv2.putText(image, 'Video Sequence Number {}'.format(sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                if results.multi_hand_landmarks and results.multi_handedness:
                    # Export keypoints that dosen't have (zeros) aka no hands
                    keypoints = extract_keypoints(results)
                    print(keypoints)
                    if not np.array_equal(keypoints , np.zeros(126)):
                        frame_list.append(keypoints)
                        frame_num +=1 
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break 
            window.append(frame_list)   
        #saving the keypoints in the action folder
        dirmax = 0
        if len(np.array(os.listdir(os.path.join(DATA_PATH, action)))) > 0:
            dirmax = len(np.array(os.listdir(os.path.join(DATA_PATH, action))))
        npy_path = os.path.join(DATA_PATH, action, str(dirmax+1))
        np.save(npy_path, window)

    cap.release()
    cv2.destroyAllWindows()   

if __name__ == '__main__':
    main()
