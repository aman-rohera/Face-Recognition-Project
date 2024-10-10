import cv2
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk,Image
import numpy as np
import os
import mysql.connector
def main():
    window=Tk()
    window.geometry('500x500+300+200')
    window.title("Face recognition system")
    window.resizable(0,0)
    l1=Label(window,text="Name",font=("times new roman",20))
    l1.grid(column=0, row=0)
    t1=Entry(window,width=50,bd=5)
    t1.grid(column=1, row=0)

    l2=Label(window,text="Age",font=("times new roman",20))
    l2.grid(column=0, row=1)
    t2=Entry(window,width=50,bd=5)
    t2.grid(column=1, row=1)

    l3=Label(window,text="Address",font=("times new roman",20))
    l3.grid(column=0, row=2)
    t3=Entry(window,width=50,bd=5)
    t3.grid(column=1, row=2)

    def train_classifier():
        data_dir="E:/Projects/Face Recognition/data"
        path = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
        faces  = []
        ids   = []
        
        for image in path:
            img = Image.open(image).convert('L');
            imageNp= np.array(img, 'uint8')
            id = int(os.path.split(image)[1].split(".")[1])
            
            faces.append(imageNp)
            ids.append(id)
        ids = np.array(ids)
        
        #Train the classifier and save
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        clf.write("classifier.xml")
        messagebox.showinfo('Result','Training dataset completed!!!')
        
    b1=Button(window,text="Training",font=("times new roman",20),bg='Rosy Brown1',fg='Black',command=train_classifier)
    b1.grid(column=0, row=4)

    def detect_face():
        def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text,clf):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)

            coords = []

            for(x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                id,pred = clf.predict(gray_image[y:y+h,x:x+w])
                confidence = int(100*(1-pred/300))
                
                mydb=mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="1234",
                database="project"
                )
                mycursor=mydb.cursor()
                mycursor.execute("select name from project where id="+str(id))
                s = mycursor.fetchone()
                s = ''+''.join(s)
                
                if confidence>74:
                    cv2.putText(img,s,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)   
                else:
                    cv2.putText(img,"UNKNOWN",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)

                coords=[x,y,w,h]
            return coords
                
        def recognize(img,clf,faceCascade):
            coords = draw_boundary(img,faceCascade,1.1,10,(255,255,255),"Face",clf)
            return img

        faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        video_capture =  cv2.VideoCapture(0)

        while True:
            ret,img = video_capture.read()
            img=  recognize(img,clf,faceCascade)
            cv2.imshow("face detection",img)

            if cv2.waitKey(1)==13:
                break

        video_capture.release()
        cv2.destroyAllWindows()

    b2=Button(window,text="Detect the face",font=("times new roman",20),bg='Dark Olive green3',fg='black',command=detect_face)
    b2.grid(column=1, row=4)

    def generate_dataset():
        if(t1.get()=="" or t2.get()=="" or t3.get()==""):
            messagebox.showinfo('Result','Please provide complete details of the user')
        else:
            mydb=mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="1234",
            database="project"
            )
            mycursor=mydb.cursor()
            mycursor.execute("SELECT * from project")
            myresult=mycursor.fetchall()
            id=1
            for x in myresult:
                id+=1
            sql="insert into project(id,Name,Age,Address) values(%s,%s,%s,%s)"
            val=(id,t1.get(),t2.get(),t3.get())
            mycursor.execute(sql,val)
            mydb.commit()
            
            face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            def face_cropped(img):
                gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray,1.3,5)
                #scaling factor=1.3
                #Minimum neighbor = 5

                if faces == ():
                    return None
                for(x,y,w,h) in faces:
                    cropped_face=img[y:y+h,x:x+w]
                return cropped_face

            cap = cv2.VideoCapture(0)
            img_id=0

            while True:
                ret,frame = cap.read()
                if face_cropped(frame) is not None:
                    img_id+=1
                    face = cv2.resize(face_cropped(frame),(200,200))
                    face  = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    file_name_path = "data/user."+str(id)+"."+str(img_id)+".jpg"
                    cv2.imwrite(file_name_path,face)
                    cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),2)
                    # (50,50) is the origin point from where text is to be written
                    # font scale=1
                    #thickness=2

                    cv2.imshow("Cropped face",face)
                    if cv2.waitKey(1)==13 or int(img_id)==100:
                        break
            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo('Result','Generating dataset completed!!!')

    b3=Button(window,text="Generate dataset",font=("times new roman",20),bg='steel blue2',fg='black',command=generate_dataset)
    b3.grid(column=2, row=4)

    window.geometry("800x200")
    window.mainloop()
def op():
    win=Tk()
    win.title('Operations')
    win.geometry('1300x800+10+10')
    win.config(bg='lightblue')
    mylbl=Label(win,text='Menu',bg='lightblue',relief='sunken',anchor='w',font=('times new roman',50)).pack()
    win.resizable(0,0)
        
    button1=Button(win,text='Insertion',bg='RosyBrown3',padx=60,pady=30,font=('times new roman',20),command=main).place(x=0,y=100,width=300)
    def modify():
        root=Tk()
        root.geometry('400x300+500+150')
        root.title('Modify Data')
            
        mylbl2=Label(root,text='Modify Record',bg='indianred3',font=('times new roman',30))
        mylbl2.pack()
        mylbl3=Label(root,text='Enter ID',font=('times new roman',20))
        mylbl3.place(x=10,y=60)
        id_entry=Entry(root,width=20)
        id_entry.place(x=250,y=70)
        mylbl4=Label(root,text='Enter Address',font=('times new roman',20))
        mylbl4.place(x=10,y=120)
        add_entry=Entry(root,width=20)
        add_entry.place(x=250,y=130)
        mylbl5=Label(root,text='Enter Age',font=('times new roman',20))
        mylbl5.place(x=10,y=190)
        age_entry=Entry(root,width=20)
        age_entry.place(x=250,y=200)
        def button__():
            if(id_entry.get()=="" or add_entry.get()=="" or age_entry.get()==""):
                messagebox.showinfo('Result','Please provide complete details of the user')

            text=id_entry.get()
            age=int(age_entry.get())
            id_=int(id_entry.get())
            mydb=mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="1234",
            database="project"
            )
            mycursor=mydb.cursor()
            mycursor.execute("select * from project where id='{}'".format(id_entry.get()))
            dat=mycursor.fetchall()
            
            if dat==[]:
                messagebox.showerror('Error',"invalid ID!!!")
            else:
                sql="update project set Address=%s,Age=%s WHERE id=%s"
                val=(add_entry.get(),age,id_)
                mycursor.execute(sql,val)
                messagebox.showinfo('Result','Data Updated!!!')
                
            if type(id_)==int:
                pass
            else:
                messagebox.showerror('Error',"Age must be in number!!!")
            mydb.commit()
        Button(root,padx=5,pady=4,font=('times new roman',15),text='Submit',command=button__).place(x=150,y=250)
        
            
    modify_btn=Button(win,text='Modification',bg='Coral2',padx=78,pady=30,command=modify,font=('times new roman',20))
    modify_btn.place(x=1000,y=100,width=300)
        
    def show():
        root=Tk()
        root.geometry('400x200+500+150')
        root.title('Show List')
        mydb=mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="1234",
            database="project"
            )
        mycursor=mydb.cursor()
        mycursor.execute("SELECT * from project")
        myresult=mycursor.fetchall()
        for i in myresult:
            mylbl1=Label(root,text=i,font=('times new roman',20)).pack(side='top',anchor='w')
    button2=Button(win,text='Display',bg='Pale Green',padx=78,pady=30,command=show,font=('times new roman',20))
    button2.place(x=0,y=350,width=300)
    def delete():
        root=Tk()
        root.geometry('400x200+500+150')
        root.title('Delete Record')
        mylbl2=Label(root,text='Delete Record',bg='hotpink4',fg='white',font=('times new roman',30))
        mylbl2.pack()
        mylbl3=Label(root,text='Enter ID',font=('times new roman',20))
        mylbl3.pack()
        a=Entry(root,width=20)
        a.pack()
        def button__():
            text=a.get()                                    #To get the value from tkinter entry box in a variable    
            print(text)
            mydb=mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="1234",
            database="project"
            )
            mycursor=mydb.cursor()
            mycursor.execute("select * from project where ID='{}'".format(int(a.get())))

            ab=mycursor.fetchall()
            if ab==[]:
                messagebox.showerror('Error',"Name Doesn't exist!!!")
            else:
                mycursor.execute("DELETE FROM project WHERE Id='{}'".format(a.get()))
                mydb.commit()
                messagebox.showinfo('Result','Data Deleted!!!')                    
        Button(root,text='Submit',command=button__).pack()
    
    button3=Button(win,text='Deletion',bg='Slate Blue3',padx=55,command=delete,pady=30,font=('times new roman',20)).place(x=1000,y=350,width=300)
    button4=Button(win,text='Exit',padx=115,pady=30,bg='Pale Violet Red',command=win.destroy,font=('times new roman',20)).place(x=520,y=570,width=300)
op()
