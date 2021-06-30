### Graduation project
#### IHearU

### How to setup:
* clone the repo 
`git clone https://github.com/Checkmate-2/IHearU.git`
* move to the the folder 
`cd .\IHearU\`
* creat virtual environment 
`python -m venv venv`
* activate the virtual environment 
`venv\Scripts\activate.ps1`
* install the requierments 
`pip install -r requirements.txt`

### How to run:
* open the terminal in the code folder
* activate the virtual environment 
`venv\Scripts\activate.ps1` if usuing powersell
or
`venv\Scripts\activate.bat` if using cmd
* run the code u want
`python collect.py`
`python train.py`
`python test.py`
## Important
* all data and model folders should be in the same folder as the code
### collect parameters
* data folder : where u wanna save the action
* action name : name of the action (a ,b , hello)
* number of sequences : number of data samples u want to collect
* number of frames per sequence : every sample has ( 2 frames ,10 frames ,30 frames) of data
* you cam source number : you camera number usualy (0) or (1) or (2)

* recording starts after 10 seconds, after the recording is done u action will be saved in the data folder


### train parameters
* data folder name : the folder that has ur actions data
* number of epochs : how many epochs u wanna train the model
* model name : name of the model if u want to save it . 
* u get a tflite model, a complete model, a csv file for the actions

### test parameters
* modelname = the model folder name 
* number of frames the model recieve : how many frames ur trained model recieve
* accuracy threshold : take a value from 0 : 1
* get highest prediction in last ... : number of predictions to consider 