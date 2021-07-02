from __future__ import unicode_literals
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score
from tensorflow import lite

def main():
    DATA_PATH = input("data folder name: ") #folder that contain actions data
    actions = np.array(os.listdir( DATA_PATH )) #list for all the actions in the data folder
    no_epochs = int(input("number of epochs: " )) #number of epochs for training
    label_map = {label:num for num, label in enumerate(actions)} #labeling actions

    # collecting the actions data
    sequences, labels = [], [] #list for all the videos and all the labels

    for action in actions:
        # find how many files in every action
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))): 
            # load files one by one for every action
            res = np.load(os.path.join(DATA_PATH, action, sequence))
            # for each frame append a lable and keypoints array
            for frame_num in range(res.shape[0]):
                labels.append(label_map[action]) 
                sequences.append(res[frame_num])  
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # Training model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1],X.shape[2])))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=no_epochs)

    # Evaluation Metrics
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    print("accuracy :"+ str(accuracy_score(ytrue, yhat)))

    # Saving Model
    modelname = input("input model name to save or leave empty to ignore : ")
    if len(modelname) > 0 :

        # Save lables as CSV
        np.savetxt(modelname+ ".txt", actions, delimiter="," , fmt='%s')
        # Saving the model
        model.save(modelname)

        # Convert the model.
        converter = lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the model as tflite.
        with open(modelname +'.tflite', 'wb') as f:
            f.write(tflite_model)

# Run Main
if __name__ == '__main__':
    main()