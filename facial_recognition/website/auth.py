from flask import Blueprint, render_template, redirect, url_for,flash
from flask import request
from . import db 
from .models import User
from .signupface import face_recog,face_recognizer
from flask_login import login_user,logout_user,login_required ,current_user
import bcrypt
from .livenessdetect import liveness
auth = Blueprint("auth", __name__)
auth._static_folder = r'C:\Users\ASUS\Desktop\facial_recognition\website\static\GIF'

@auth.route("/login",methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        face_embedding = face_recog()
        
        if user:
            if bcrypt.checkpw(password.encode(), user.password):
                if face_embedding != []:
                    if face_recognizer(face_embedding,user.face_embedding):
                        if liveness():
                            flash("login successful")
                            login_user(user,remember=True)
                            return redirect(url_for('views.home'))
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
    return render_template("login.html")


@auth.route("/sign-up",methods=['GET','POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        username_exists = User.query.filter_by(username=username).first()
        email_exits = User.query.filter_by(email=email).first()
        face_embedding = face_recog()
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
        else:
            new_User = User(email=email,username=username,password=bcrypt.hashpw(password1.encode(), bcrypt.gensalt()),face_embedding=str(face_embedding))

            db.session.add(new_User)
            db.session.commit()
            login_user(new_User,remember=True)
            flash('User created',category='success')
            return redirect(url_for('views.home'))
    return render_template("signup.html")

@auth.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("views.home"))

