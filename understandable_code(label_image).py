import cv2
import label_image

size = 4   # size haie yeh, jo resize wageyra krne mein help krta hai


classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# haar ek pre-trained xml file hai, jo opencv wale hi dete hai


webcam = cv2.VideoCapture(0) 
# direct webcamera se videocapture

while True :
    (rval, im) = webcam.read() #rval :: read value from webcamera
    im = cv2.flip(im,1,0) # camera ko flip kr diya taaki, woh mirror jaise kaam kare
    
    mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)) )
    # yaha resize kiya gya 1/4 mein, taaki detection jaldi ho
    
    faces = classifier.detectMultiscale(mini)
    # multi-scale matlab :: face
    # face detect hoga : classifier ke help se :: xml file jo humne laod ki
    
    for f in faces :
        (x, y, w, z) = [v*size for v in f]
        # co-ordinates hai x,y,w,z :: otline around face
        # v*size : kiye :: to resize the 1/4 shape , back into orginal shape
        
        cv2.rectangle(im, (x,y), (x+w, y+h), (0,255,0) ,4 )
        # draw rectanle around the face
        
        sub_face = im[y:y+h,  x:x+w]
        # ek frame ko save krne ke liye (out of all live record frames)
        
        FaceFileName = "test.jpg"
        # saving current image from the webcam
        
        cv2.imwrite(FaceFileName, sub_face)
        # Facefile image ko ,
        # sub_face folder mein dave krna
        
        text = label_image.main(FaceFileName)
        # getting result from label_image file :: classification results
        text = text.title()
        font = cv2.FONT_HERSHAY_TRIPLEX
        cv2.putText(im, text, (x+w, y), font, 1, (0,0,255), 2 )
        
    cv2.imShow('Capture', im)
    key = cv2.waitkey(10)
    
    if key==27:
        break # if esc key pressed, break out.
    
    