from flask import Blueprint, render_template, redirect, url_for,flash
from flask import request
from . import db 
import pyttsx3
from .models import User
from .signupface import face_recog,face_recognizer
from flask_login import login_user,logout_user,login_required ,current_user
import bcrypt
from .livenessdetect import liveness
from .hand import create,compare
import cv2
import warnings
import time
warnings.filterwarnings("ignore")



auth = Blueprint("auth", __name__)
auth._static_folder = r'..\website\static\GIF'     

@auth.route("/login",methods=['GET','POST'])
def login():
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_HW_ACCELERATION, 1.0)
    if request.method == 'POST':
        engine = pyttsx3.init()
        email = request.form.get('email')
        #password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        #flash("Please wait your being recognised")
        engine.say("Please wait,Your Face is being recognised")
        engine.runAndWait()
        start_time = time.time()
        flag , face_embedding = liveness(vid,2)
        end_time = time.time()

        print("Time taken Sign in using both face & live:", end_time - start_time, "seconds")
        if user:
            if True: #bcrypt.checkpw(password.encode(), user.password):
                if face_embedding != []:
                    if face_recognizer(face_embedding,user.face_embedding):
                        #flash("Please wait your face being recognised")
                        
                        
                        if flag:
                            #flash("Please show your right hand")
                            engine.say("Please show your right hand")
                            engine.runAndWait()
                            start_time = time.time()
                            try:
                                hand_module = create(vid)
                            except:
                                hand_module = None
                            end_time = time.time()

                            print("Time taken Hand sign in:", end_time - start_time, "seconds")
                            if hand_module is not None:
                                if compare(hand_module,user.hand_model):
                                    flash("login successful")
                                    engine.say("Thank you,You Have successfully signed in")
                                    engine.runAndWait()
                                    login_user(user,remember=True)
                                    return redirect(url_for('views.home'))
                                else:
                                    flash("Hand Missmatch")
                            else:
                                flash('Hand is not recoded')
                        else:
                            flash("Liveness failed")
                    else:
                        flash("Face Mismatch")
                else:
                    flash("Face is not Detected")
            else:
                flash("Invalid Password")
        else:
            flash("Invalid Email")
    vid.release()
    cv2.destroyAllWindows()
    return render_template("login.html")


@auth.route("/sign-up",methods=['GET','POST'])
def sign_up():
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_HW_ACCELERATION, 1.0)
    if request.method == 'POST':
        engine = pyttsx3.init()
        email = request.form.get('email')
        username = request.form.get('username')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        username_exists = User.query.filter_by(username=username).first()
        email_exits = User.query.filter_by(email=email).first()
        engine.say("Please wait! System is capturing your face")
        engine.runAndWait()
        start_time = time.time()
        face_embedding = face_recog(vid)
        end_time = time.time()
        print("Time taken to recog in sign up:", end_time - start_time, "seconds")
        engine.say("Please show your right hand")
        engine.runAndWait()
        #flash("Please show your right hand")
        start_time = time.time()
        hand_module = create(vid)
        end_time = time.time()
        print("Time taken to hand recog in sign up:", end_time - start_time, "seconds")
        if email_exits:
            flash('Email is already in use',category="error")
        elif username_exists:
            flash('Username is already in use',category="error")
        elif password1 != password2: 
            flash('Passwords do not match',category="error")
        elif len(username)<2:
            flash('Username must be atleast 2 characters',category="error")
        elif len(password1)<4:
            flash('Password must be atleast 4 characters',category="error")
        elif face_embedding == []:
            flash('Face not detected',category="error")
        elif hand_module is None:
            flash("Hand is not detected",category="error")
        else:
            new_User = User(email=email,username=username,password=bcrypt.hashpw(password1.encode(), bcrypt.gensalt()),face_embedding=str(face_embedding),hand_model=str(hand_module))
            db.session.add(new_User)
            db.session.commit()
            login_user(new_User,remember=True)
            flash('User created',category='success')
            engine.say("Thank you,You Have successfully signed in")
            engine.runAndWait()
            return redirect(url_for('views.home'))
    
    return render_template("signup.html")

@auth.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("views.home"))

