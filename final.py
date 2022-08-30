def collect():
    import cv2
    import numpy as np

    #cascade classifier =  helps the machine to identify face

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    def face_extractor(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)


        if faces is None:
            return None
        
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        

            return cropped_face

    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret,frame = cap.read()
        if face_extractor(frame) is not None :
          count = count+1
          face = cv2.resize(face_extractor(frame),(200,200))

          face = cv2.cvtColor(face , cv2.COLOR_BGR2GRAY)


          file_name_path = 'D:/facefolder/potia'+str(count)+'.jpg'
          cv2.imwrite(file_name_path,face)
          cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
          cv2.imshow('Face Cropper',face)
        else:
            print("face not found")
            pass


        if cv2.waitKey(1) == 13:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("collected")











from tkinter import *
root = Tk()
from PIL import Image,ImageTk


def fa():
    
        import cv2
        import numpy as np
        from os import listdir
        from os.path import isfile, join 

        data_path = 'D:/facefolder/'
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

        Training_Data, Labels = [], []

        for i, files in enumerate(onlyfiles):
            image_path = data_path + onlyfiles[i]
            images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(i)

        labels = np.asarray(Labels, dtype=np.int32)

        model = cv2.face.LBPHFaceRecognizer_create()

        model.train(np.asarray(Training_Data), np.asarray(Labels))

        print("Model Training Complete")


        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        def face_detector(img, size = 0.5):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)
            if faces is():
                return img,[]

            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
                roi = img[y:y+h, x:x+w]
                roi = cv2.resize(roi,(200,200))
            

            return img,roi

        cap = cv2.VideoCapture(0)

        while True:


            ret, frame = cap.read()
            image, face = face_detector(frame)
            try:
               face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
               result = model.predict(face)

               if result[1] < 500:
                chances = int(100*(1-(result[1])/300))
                display_string = str(chances)+"  %   chances of being user"

               cv2.putText(image,display_string,(100,120) , cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255))

               if chances >84:
                 cv2.putText(image,"unlocked" ,(250,450), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                 cv2.imshow('Face Cropper',image)
            

               else:
                cv2.putText(image,"locked" ,(250,450), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
               
                cv2.imshow('Face Cropper',image)
                


            
            
            except:
                cv2.putText(image,"face not found" ,(250,450), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                
                cv2.imshow('Face Cropper',image)
                pass


            if cv2.waitKey(1) == 13:
                break

        cap.release()
        cv2.destroyAllWindows()



root.geometry("800x500")
root.title("Face Recogniser")

image = Image.open("photo.jpg")
photo = ImageTk.PhotoImage(image)
show = Label(image=photo)
show.pack()

b = Button(text="Face Verification",command=fa,font=('Times New Roman',15,'bold'),fg="red",bg="black")
k = Button(text="Collect Samples",command=collect,font=('Times New Roman',15,'bold'),fg="red",bg="black")
k.place(x=420,y=455)

b.place(x=250, y=455)



root.mainloop()

