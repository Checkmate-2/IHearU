from __future__ import unicode_literals
import cv2
import numpy as np
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections 

def draw_styled_landmarks(image, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
# get left hand and right hand landmarks if there.
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
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
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop through actions
        cv2.waitKey(5000) #at the start wait 2 seconds
        window = [] # the list that will contain all the landmarks
        # Loop through sequences aka videos
        for sequence in range(no_sequences):   
            frame_list = []
            frame_num = 0
            # Loop through video length aka number of frames
            while frame_num < no_frames: 
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # Apply wait logic
                if sequence == 0: 
                    cv2.putText(image, 'STARTING COLLECTION IN 2s', (120,200), 
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
                
                # Export keypoints that dosen't have (zeros) aka no hands
                keypoints = extract_keypoints(results)
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
            dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
        npy_path = os.path.join(DATA_PATH, action, str(dirmax+1))
        np.save(npy_path, window)
                        
    cap.release()
    cv2.destroyAllWindows()   

if __name__ == '__main__':
    main()