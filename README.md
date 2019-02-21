# Facial-Expression-Detection


Facial Expression can be used to know whether a person is sad, happy, angry expression.


# DEPENDENCIES

Install this dependencies first :

    pip install tensorflow
    pip install opencv-python
    

## STEP 1 - Implementation of OpenCV 'haarcascade' :

I'm using the "Frontal Face Alt" Classifier for detecting the presence of Face in the WebCam.

    classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

- Its a pre trained model, by opencv itself to help you with face recognisation.



## STEP 2 - ReTraining  :

- We need to first create a directory named images like Happy, Sad, Angry, Calm and Neutral etc.
- Now fill these directories with respective images by downloading them from the any dataset or google-image. 

- Now run the "face-crop.py" and update you saved image's directory. 
- Now, run this code in cmd or terminal :

      python retrain.py --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --architecture=MobileNet_1.0_224 --image_dir=images

It will iterate, and will take some time to train the model, like 15-20 mins.



## STEP 3 - Importing the ReTrained Model :

- Now run the "label.py" program by typing the following in cmd or in Terminal:
      
     python label.py
     
It'll open a new window of OpenCV and then identifies your Facial Expression.

NOTE : I took very less number of images to train, so , it results into appx 66% accuracy.
I suggest you to train with atleast 150-200 images to get 85+% accuracy. 
Thanks!
