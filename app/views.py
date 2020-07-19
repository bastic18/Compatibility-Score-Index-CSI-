"""
Flask Documentation:     http://flask.pocoo.org/docs/
Jinja2 Documentation:    http://jinja.pocoo.org/2/documentation/
Werkzeug Documentation:  http://werkzeug.pocoo.org/documentation/
This file creates your application.
"""
import os
import math

from app import app, login_manager
from flask_mysqldb import MySQL
from flask import render_template, request, redirect, url_for, flash, session, jsonify
from app.forms import LoginForm, SignUp, Groupings, newSet, joinNewSet, AboutYou, Criteria, Profile_About, GroupNum, AboutFriend, adminSettings, TranferGrp
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from app import CSI_code
import random
import uuid
from decimal import Decimal
import json
import simplejson as json
import secrets
from PIL import Image

import smtplib
from email.message import EmailMessage
from pathlib import Path
from string import Template
# import simplejson as json

###
# Routing for your application.
###

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'csi_capstone'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)


@app.route('/')
def home():
    """Render website's home page."""
    if 'logged_in' in session:
        mycursor = mysql.connection.cursor()
        mycursor.execute(
            'Select * from Biography WHERE user_id = %s', (session['id'],))
        biography = mycursor.fetchone()
        return render_template('home.html', biography=biography)
    return render_template('home.html')


@app.route('/about/')
def about():
    """Render the website's about page."""
    if 'logged_in' in session:
        mycursor = mysql.connection.cursor()
        mycursor.execute(
            'Select * from Biography WHERE user_id = %s', (session['id'],))
        biography = mycursor.fetchone()
        return render_template('about.html', biography=biography)
    return render_template('about.html')


@app.route('/testimonies/')
def testimonies():
    """Render the website's about page."""
    if 'logged_in' in session:
        mycursor = mysql.connection.cursor()
        mycursor.execute(
            'Select * from Biography WHERE user_id = %s', (session['id'],))
        biography = mycursor.fetchone()
        return render_template('testimonies.html', biography=biography)
    return render_template('testimonies.html')


@app.route('/admin/', methods=["GET", "POST"])
def admin():
    """Render the website's admin page."""
    form = adminSettings()

    mycursor = mysql.connection.cursor()

    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()

    mycursor.execute('Select * from Dictionary')
    definitions = mycursor.fetchone()

    if request.method == "POST" and form.validate_on_submit():
        pers_weight = request.form['pers_weight']
        ldrshp_weight = request.form['ldrshp_weight']
        hobby_weight = request.form['hobby_weight']
        democratic = request.form['democratic']
        autocratic = request.form['autocratic']
        laissezfaire = request.form['laissezfaire']
        ambivert = request.form['ambivert']
        extrovert = request.form['extrovert']
        introvert = request.form['introvert']
        sports = request.form['sports']
        music = request.form['music']
        exercising = request.form['exercising']
        reading = request.form['reading']
        shopping = request.form['shopping']
        writing = request.form['writing']
        dancing = request.form['dancing']
        arts = request.form['arts']
        watchingTV = request.form['watchingTV']

        mycursor = mysql.connection.cursor()

        if pers_weight:
            sql = "UPDATE Dictionary SET personality_weight = %s  WHERE dict_id = %s"
            val = (pers_weight, 'D-01')
            mycursor.execute(sql, val)
            mysql.connection.commit()

        if ldrshp_weight:
            sql = "UPDATE Dictionary SET leadership_weight = %s WHERE dict_id = %s"
            val = (ldrshp_weight, 'D-01')
            mycursor.execute(sql, val)
            mysql.connection.commit()

        if hobby_weight:
            sql = "UPDATE Dictionary SET hobby_weight = %s WHERE dict_id = %s"
            val = (hobby_weight, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if democratic:
            sql = "UPDATE Dictionary SET democratic = %s WHERE dict_id = %s"
            val = (democratic, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if autocratic:
            sql = "UPDATE Dictionary SET autocratic = %s WHERE dict_id = %s"
            val = (autocratic, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if laissezfaire:
            sql = "UPDATE Dictionary SET laissez_faire = %s WHERE dict_id = %s"
            val = (laissezfaire, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if ambivert:
            sql = "UPDATE Dictionary SET ambivert = %s WHERE dict_id = %s"
            val = (ambivert, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if extrovert:
            sql = "UPDATE Dictionary SET extrovert = %s WHERE dict_id = %s"
            val = (extrovert, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if introvert:
            sql = "UPDATE Dictionary SET introvert = %s WHERE dict_id = %s"
            val = (introvert, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if sports:
            sql = "UPDATE Dictionary SET sports = %s WHERE dict_id = %s"
            val = (sports, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if music:
            sql = "UPDATE Dictionary SET music = %s WHERE dict_id = %s"
            val = (music, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if exercising:
            sql = "UPDATE Dictionary SET exercising = %s WHERE dict_id = %s"
            val = (exercising, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if reading:
            sql = "UPDATE Dictionary SET reading = %s WHERE dict_id = %s"
            val = (reading, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if shopping:
            sql = "UPDATE Dictionary SET shopping = %s WHERE dict_id = %s"
            val = (shopping, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if writing:
            sql = "UPDATE Dictionary SET writing = %s WHERE dict_id = %s"
            val = (writing, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if dancing:
            sql = "UPDATE Dictionary SET dancing = %s WHERE dict_id = %s"
            val = (dancing, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if arts:
            sql = "UPDATE Dictionary SET arts = %s WHERE dict_id = %s"
            val = (arts, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        if watchingTV:
            sql = "UPDATE Dictionary SET watching_tv = %s WHERE dict_id = %s"
            val = (watchingTV, 'D-01')

            mycursor.execute(sql, val)
            mysql.connection.commit()

        flash('Settings Updated', 'success')

    return render_template('admin.html', form=form, definitions=definitions, biography=biography)


@app.route('/admin/currentusers/', methods=["GET", "POST"])
def allUsers():
    mycursor = mysql.connection.cursor()
    mycursor.execute('Select * from user WHERE user_id != %s',
                     (session['id'],))
    users = mycursor.fetchall()

    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()

    return render_template('all_users.html', users=users, biography=biography)


@app.route('/deleteuser/<user_id>')
def deleteUser(user_id):
    mycursor = mysql.connection.cursor()

    sql = "DELETE from User WHERE user_id = %s"
    val = (user_id,)

    mycursor.execute(sql, val)
    mysql.connection.commit()

    flash('User has been removed', 'success')
    return redirect(url_for("allUsers"))


@app.route("/login", methods=["GET", "POST"])
def login():
    mycursor = mysql.connection.cursor()
    form = LoginForm()

    if 'logged_in' in session:
        mycursor.execute(
            'Select * from Biography WHERE user_id = %s', (session['id'],))
        biography = mycursor.fetchone()
        return redirect(url_for('home'))

    if request.method == "POST" and form.validate_on_submit():
        username = form.username.data
        # Query if User exists
        mycursor.execute(
            'SELECT * FROM user WHERE username = %s AND password = %s', (username, form.password.data,))
        user = mycursor.fetchone()

        # If existing user
        if user:
            # Create session data, we can access this data in other routes
            session['logged_in'] = True
            session['id'] = user['user_id']
            session['username'] = request.form['username']
            session['TYPE'] = user['type']
            session['first_name'] = user['first_name']
            session['last_name'] = user['last_name']

            if user['type'] == 'Administrator':
                return redirect(url_for('admin'))

            else:
                flash('Login Successful', 'success')
                return redirect(url_for('dashboard', username=username))

        # Flash error message with incorrect username/password
        flash('Invalid Credentials', 'danger')
    return render_template("login.html", form=form)


@app.route('/dashboard/<username>', methods=["GET", "POST"])
def dashboard(username):
    """Render the website's dashboard page."""
    if 'username' in session:
        mycursor = mysql.connection.cursor()
        mycursor.execute('Select * from user WHERE username = %s', (username,))
        user = mycursor.fetchone()

        if session.get('TYPE') == "Organizer":
            mycursor.callproc('GetSetCount', (session['id'], ))
            for result in mycursor:
                result = result['count(distinct sid)']
            mycursor.execute(
                'SELECT * FROM sets WHERE organizer = %s ', (session['id'],))
        else:
            mycursor.callproc('GetGroupCount', (session['id'], ))
            for result in mycursor:
                result = result['count(distinct sid)']
            mycursor.execute(
                'SELECT * FROM Sets JOIN joinset ON sets.sid = joinset.sid WHERE joinset.user_id = %s ', (session['id'],))

        getSets = mycursor.fetchall()

        if session.get('TYPE') == "Regular":
            mycursor.execute(
                'SELECT * FROM user JOIN pin_user ON user.user_id = pin_user.match_id WHERE pin_user.user_id = %s ', (session['id'],))
            getFriends = mycursor.fetchall()

        mycursor.execute(
            'Select * from Biography WHERE user_id = %s', (session['id'],))
        biography = mycursor.fetchone()

    return render_template('dashbrd.html', groups=getSets, type=session.get('TYPE'), biography=biography)

def save_picture(form_picture):
    random_hex= secrets.token_hex(8)
    f_name,f_ext= os.path.splitext(form_picture.filename)
    picture_fn= random_hex+ f_ext
    # picture_path= os.path.join(app.root_path,'static/uploads',picture_fn)
    # output_size=(450,550)
    # i=Image.open(form_picture)
    # i.thumbnail(output_size)
    # i.save(picture_path)
    return picture_fn


@app.route("/edit/<username>", methods=["GET", "POST"])
def edit_info(username):
    PropicForm = Profile_About()
    mycursor = mysql.connection.cursor()

    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()
    filename=''
    if request.method == "POST" and PropicForm.validate_on_submit():
        profPic = request.files['profPic']
        about = PropicForm.about.data
        
        # filename = secure_filename(profPic.filename)
        # print("filename>>>>>>>>>>>", filename)
        if profPic.filename != "":
            filename = save_picture (profPic)
        else:
            filename=biography['pro_photo']

        # random_hex= secrets.token_hex(12)
        # f_name,f_ext= os.path.splitext(profPic.filename)
        # picture_fn= random_hex+ f_ext
        # filename = secure_filename(picture_fn)
        # print("filename>>>>>>>>>>>", filename)

        if profPic.filename != "":  # INCASE THEY RE-EDIT AND DONT UPLOAD A PHOTO
            profPic.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        if biography != None:
            if profPic != "" and about != "":  # IF PERSON UPDATES BOTH
                sql = "UPDATE Biography SET pro_photo = %s, about = %s WHERE user_id = %s"
                val = (filename, about, session['id'])
            elif profPic != "":  # IF PERSON UPDATES PHOTO ONLY
                sql = "UPDATE Biography SET pro_photo = %s WHERE user_id = %s"
                val = (filename, session['id'])
            else:  # IF PERSON UPDATES ABOUT ONLY
                sql = "UPDATE Biography SET about = %s WHERE user_id = %s"
                val = (about, session['id'])

            mycursor.execute(sql, val)
            mysql.connection.commit()

        else:  # EDIT FOR THE FIRST TIME

            sql = "INSERT INTO Biography (user_id, pro_photo, about) VALUES (%s, %s, %s)"
            val = (session['id'], filename, about)
            mycursor.execute(sql, val)
            mysql.connection.commit()

        flash('Your edits were saved', 'success')
        redirect(url_for('dashboard',  username=session.get('username')))

    return render_template('edit.html', PropicForm=PropicForm, biography=biography)


@app.route("/logout")
def logout():
    # Logout the user and end the session
    session.pop('logged_in', None)
    session.pop('id', None)
    session.pop('username', None)
    session.pop('type', None)
    session.pop('first_name', None)
    session.pop('last_name', None)
    return redirect(url_for('home'))


@app.route('/registerAs', methods=["GET", "POST"])
def registerAs():
    """Render the website's register page."""
    if request.method == "POST":
        if request.form.get('Regular') == 'Regular':
            return redirect(url_for('register', typeUser="Regular"))

        elif request.form.get('Organizer') == 'Organizer':
            return redirect(url_for('register', typeUser="Organizer"))
    return render_template('registerAs.html')


@app.route('/register/<typeUser>', methods=["GET", "POST"])
def register(typeUser):
    # First Name, Last Name, Email, Password and Username are collected from the SignUp Form
    form = SignUp()

    if request.method == "POST" and form.validate_on_submit():
        mycursor = mysql.connection.cursor()
        # Collects username and email info from form
        username = form.username.data
        email = form.email.data

        # Checks if another user has this username
        mycursor.execute(
            'SELECT * FROM user WHERE username = %s', (username,))
        existing_username = mycursor.fetchone()

        # Checks if another user has this email address
        mycursor.execute(
            'SELECT * FROM user WHERE email = %s', (email,))
        existing_email = mycursor.fetchone()

        # Gets last record in User Table
        mycursor.execute(
            'SELECT * from user ORDER BY user_id DESC LIMIT 1')
        lastRec = mycursor.fetchone()
        if lastRec is None:
            lastRec = 1
        else:
            lastRec = lastRec['user_id'] + 1

        # If unique email address and username provided then log new user
        if existing_username is None and existing_email is None:
            mycursor = mysql.connection.cursor()
            sql = "INSERT INTO User (user_id, type, first_name, last_name, username, email, password) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            val = (lastRec, typeUser, request.form['fname'], request.form['lname'],
                   request.form['username'], request.form['email'], request.form['password'])

            mycursor.execute(sql, val)
            mysql.connection.commit()

            last = mycursor.lastrowid

            # Specialisation of Users
            if typeUser == "Regular":
                # Calls RandomFeatures  function to generate features for regular user
                randFt = randomFeatures()

                sql = "INSERT INTO Regular (user_id, sex, age, height, leadership, ethnicity, personality, education, hobby, occupation, pref_sex, pref_ethnicity) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                val = (mycursor.lastrowid, randFt[0], randFt[1],
                       randFt[2], randFt[3], randFt[4], randFt[5], randFt[6], randFt[7], randFt[8], randFt[9], randFt[10])

                mycursor.execute(sql, val)
                mysql.connection.commit()

            else:
                position = random.choice(
                    ['Secretary', 'CEO', 'Treasurer', 'Vice President', 'Supervisor', 'Manager'])

                sql = "INSERT INTO Organizer (user_id, position) VALUES (%s, %s)"
                val = (mycursor.lastrowid, position)

                mycursor.execute(sql, val)
                mysql.connection.commit()

            # Success Message Appears
            flash('Successfully registered', 'success')

            # Logs in a newly registered user
            session['logged_in'] = True
            session['id'] = last
            session['username'] = request.form['username']
            session['TYPE'] = typeUser
            session['first_name'] = request.form['fname']
            session['last_name'] = request.form['lname']

            # Redirects to Profile Page
            return redirect(url_for('dashboard', username=session.get('username')))

    # Flash errors in form and redirects to Register Form
    flash_errors(form)
    return render_template("signup.html", form=form)


@app.route('/<username>/createSet',  methods=['GET', 'POST'])
def createSet(username):
    """Render the website's page."""
    form = newSet()
    mycursor = mysql.connection.cursor()
    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()
    mycursor.execute('Select email from User WHERE username = %s', (username,))
    curr_email=mycursor.fetchone()
    
    # If user is logged in session and is an organizer
    if 'username' in session and session.get('TYPE') == "Organizer":

        if request.method == "POST" and form.validate_on_submit():
            set_name = form.set_name.data
            purpose = form.purpose.data
            cc_email=form.email.data

            mycursor.execute(
                'SELECT * FROM sets WHERE set_name = %s', (set_name,))
            existing_set = mycursor.fetchone()

            if existing_set is None:
                # Get last index in Sets Relation
                mycursor.execute(
                    'SELECT * from sets ORDER BY sid DESC LIMIT 1')
                lastRec = mycursor.fetchone()
                if lastRec is None:
                    lastRec = 1
                else:
                    lastRec = lastRec['sid'] + 1

                # Generates Random Code for the Set
                code = uuid.uuid4().hex.upper()[0:10]

                #######codeeeeee email
                html= Template(Path('app\email.html').read_text())
                email= EmailMessage()
                email['from']= 'Teamcsi Capstone'
                email['to']= curr_email['email']
                
                #email['cc']= "bastic.grant88@gmail.com, carl.beckford@uwimona.edu.jm,lanainevers@gmail.com, hc.mccalla@gmail.com"
                email['cc']= cc_email
                email['subject']= 'TEAM CSI CAPSTONE PROJECT'

                email.set_content(html.substitute({'name': set_name, 'purpose': purpose, 'code':code}), 'html')

                with smtplib.SMTP(host='smtp.gmail.com', port=587) as smtp:
                    smtp.ehlo()
                    smtp.starttls()
                    smtp.login('teamcsi.capstone@gmail.com','capCSI2020')
                    smtp.send_message(email)
                    print("Email successfully sent")

                


                # Insert a New Set for an Organizer
                sql = "INSERT INTO Sets (sid, set_name, purpose, code, organizer) VALUES (%s, %s, %s, %s, %s)"
                val = (lastRec, set_name, purpose, code, session.get('id'))

                mycursor.execute(sql, val)
                mysql.connection.commit()

                # Success Message Appears
                flash('Set Added', 'success')

                # Redirects to Dashboard
                return redirect(url_for('dashboard', username=session.get('username')))
    return render_template('createSet.html', form=form, biography=biography)


@app.route('/<username>/joinSet',  methods=['GET', 'POST'])
def joinASet(username):
    """Render the website's  page."""
    form = joinNewSet()
    mycursor = mysql.connection.cursor()
    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()
    if request.method == "POST" and form.validate_on_submit() and session.get('TYPE') == "Regular":
        # Code entered in form for joining a set
        code = form.set_code.data

        # Checks if entered code is valid
        mycursor.execute(
            'SELECT * FROM sets WHERE code = %s', (code,))
        existing_code = mycursor.fetchone()
        # If valid credentials, flash success and redirect
        if existing_code:
            mycursor = mysql.connection.cursor()
            sql = "INSERT INTO joinSet (user_id, sid) VALUES (%s, %s)"
            val = (session.get('id'), existing_code['sid'])
            mycursor.execute(sql, val)
            mysql.connection.commit()

            flash('Successfully Added to Set', 'success')

            return redirect(url_for('dashboard', username=session.get('username')))
    return render_template('joinSet.html', form=form, biography=biography)


@app.route('/members/<sid>',  methods=['GET', 'POST'])
def members(sid):


    form = Groupings()
    mycursor = mysql.connection.cursor()
    mycursor.execute('SELECT * from sets WHERE sid = %s', (sid,))
    fullSet = mycursor.fetchall()
    fresults=[]
    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()

    if 'username' in session and session.get('TYPE') == "Regular":

        mycursor.execute('SELECT set_name FROM Sets WHERE sid = %s', (sid,))
        set_name = mycursor.fetchone()

        mycursor.execute(
            'SELECT group_num FROM SetUserGroup WHERE sid = %s AND username = %s', (sid, session['username'], ))
        group = mycursor.fetchone()
        
        if group:  # if the regular user is in groups
            mycursor.execute(
                'SELECT first_name, last_name, user.username, leader from User JOIN SetUserGroup on SetUserGroup.username = user.username WHERE group_num = %s and sid=%s', (group['group_num'],sid, ))
            getMembers = mycursor.fetchall()
            
            mycursor.execute(
                'SELECT sum(CSI) as csi, sum(percentage) as percentage, sum(personality_score) as personality_score, sum(leadership_score) as leadership_score, sum(hobby_score) as hobby_score, sum(gender_score) as gender_score, sum(age_score) as age_score, sum(height_score) as height_score, sum(ethnicity_score) as ethnicity_score, sum(education_score) as education_score, sum(occupation_score) as occupation_score, count(CSI) as amount from SetUserGroup JOIN SetGroupScore ON SetGroupScore.`userA username` = SetUserGroup.username AND SetGroupScore.group_num = SetUserGroup.group_num WHERE SetUserGroup.sid = %s AND SetGroupScore.group_num = %s', (sid, group['group_num'], ))
            cur_set = mycursor.fetchall()
            print('currrrr set>>>>>>>>>', cur_set)
            return render_template('members.html', grp_num=group['group_num'], set_name=set_name, biography=biography, getMembers=getMembers)
        else:  # if they aren't
            return render_template('members.html', set_name=set_name, biography=biography)

    # FOR ORGANIZER SIDE
    if 'username' in session and session.get('TYPE') == "Organizer":
        mycursor.execute('SELECT user.user_id, username, first_name, last_name, sex, pref_sex, age, height, leadership, education, ethnicity, pref_ethnicity, hobby, occupation, personality from user JOIN regular JOIN joinset ON regular.user_id = user.user_id and regular.user_id = joinset.user_id and user.user_id=joinset.user_id WHERE joinset.sid = %s', (sid,))
        getMembers = list(mycursor.fetchall())

        mycursor.execute('SELECT * from setusergroup WHERE sid = %s', (sid,))
        formed = mycursor.fetchall()

        if formed:
            mycursor.execute(
                'SELECT max(group_num)from SetUserGroup where sid = %s ', (sid, ))
            grpAmt = list(mycursor.fetchall())
            return render_template('miniGrps.html', fullSet=fullSet[0], grpAmt=grpAmt[0]['max(group_num)'], biography=biography)

        # if request.method == "POST" and form.validate_on_submit() and int(form.numPersons.data) <= len(getMembers):
        # print('ylo',len(getMembers) )
        if request.method == "POST" and form.validate_on_submit() and int(form.numPersons.data) <= len(getMembers):
            grpBy = form.grpBy.data
            numPersons = form.numPersons.data
            leadership= form.leadership.data
            personality=form.personality.data
            hobby=form.hobby.data
            height=form.height.data
            age=form.age.data
            gender=form.gender.data
            ethnicity=form.ethnicity.data
            education=form.education.data
            occupation=form.occupation.data
            print('critr>>>>>>>>>>>>>>>',grpBy)
            print('leadership>>>>>>>>>>>>>>>',leadership)
            print('p>>>>>>>>>>>>>>>',personality)
            print('h>>>>>>>>>>>>>>>',hobby)
            print('he>>>>>>>>>>>>>>>',height)
            print('a>>>>>>>>>>>>>>>',age)
            print('gender>>>>>>>>>>>>>>>',gender)
            print('e>>>>>>>>>>>>>>>',ethnicity)
            print('ed>>>>>>>>>>>>>>>',education)
            print('occ>>>>>>>>>>>>>>>',occupation)

            if grpBy=="compatible":   
                print('yolo')       
                results= CSI_code.GRP_SETUP3(getMembers,int(numPersons), float(leadership), float(personality), float(hobby),float(height),
                                                float(age), float(gender), float(ethnicity), float(education), float(occupation))
                print('polo')
                grpAmt= len(results)
                fresults.append(results)
                fresults=fresults[0]
                
                
                mycursor.execute(
                'DELETE from SetUserGroup WHERE sid = %s', (sid,))

                mycursor.execute(
                'DELETE from SetGroupScore WHERE sid = %s', (sid,))

                for i in range(grpAmt):
                    grpnum=i+1
                 
                    sql = "INSERT INTO SetUserGroup (username, sid, group_num, leader) VALUES (%s, %s, %s, %s)"
                    val = (fresults[i][0]['userA username'], sid,grpnum, 1 )

                    mycursor.execute(sql, val)
                    mysql.connection.commit()
                    for result in fresults[i]:
                    
                        sql = "INSERT INTO SetUserGroup (username, sid, group_num) VALUES (%s, %s, %s)"
                        val = (result['userB username'], sid, i+1)

                        mycursor.execute(sql, val)
                        mysql.connection.commit()

                        sql = "INSERT INTO SetGroupScore (sid, group_num,`userA username`,`userB username`, CSI, percentage, personality_score, leadership_score, hobby_score, gender_score, age_score, height_score, ethnicity_score, education_score, occupation_score, con_personality_score, con_leadership_score, con_hobby_score, con_gender_score, con_age_score, con_height_score, con_ethnicity_score, con_education_score, con_occupation_score) VALUES (%s, %s,%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

                        val = (sid, grpnum, result['userA username'],result['userB username'], result['CSI'], result['Percentage'], result['personality_score'], result['leadership_score'], result['hobby_score'], result['gender_score'], result['age_score'], result['height_score'], result['ethnicity_score'], result['education_score'],
                            result['occupation_score'], result['con_personality_score'], result['con_leadership_score'], result['con_hobby_score'], result['con_gender_score'], result['con_age_score'], result['con_height_score'], result['con_ethnicity_score'], result['con_education_score'], result['con_occupation_score'])

                        mycursor.execute(sql, val)
                        mysql.connection.commit()


            if grpBy=="uncompatible":          
                results= CSI_code.GRP_SETUP4(getMembers,int(numPersons), float(leadership), float(personality), float(hobby),float(height),
                                                float(age), float(gender), float(ethnicity), float(education), float(occupation))
                grpAmt= len(results)
                fresults.append(results)
                fresults=fresults[0]

                mycursor.execute(
                'DELETE from SetUserGroup WHERE sid = %s', (sid,))

                mycursor.execute(
                'DELETE from SetGroupScore WHERE sid = %s', (sid,))

                for i in range(grpAmt):
                    grpnum=i+1
                 
                    sql = "INSERT INTO SetUserGroup (username, sid, group_num, leader) VALUES (%s, %s, %s, %s)"
                    val = (fresults[i][0]['userA username'], sid,grpnum, 1)

                    mycursor.execute(sql, val)
                    mysql.connection.commit()
                    for result in fresults[i]:
                    
                        sql = "INSERT INTO SetUserGroup (username, sid, group_num) VALUES (%s, %s, %s)"
                        val = (result['userB username'], sid, i+1)

                        mycursor.execute(sql, val)
                        mysql.connection.commit()

                        sql = "INSERT INTO SetGroupScore (sid, group_num,`userA username`,`userB username`, CSI, percentage, personality_score, leadership_score, hobby_score, gender_score, age_score, height_score, ethnicity_score, education_score, occupation_score, con_personality_score, con_leadership_score, con_hobby_score, con_gender_score, con_age_score, con_height_score, con_ethnicity_score, con_education_score, con_occupation_score) VALUES (%s, %s,%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

                        val = (sid, grpnum, result['userA username'],result['userB username'], result['CSI'], result['Percentage'], result['personality_score'], result['leadership_score'], result['hobby_score'], result['gender_score'], result['age_score'], result['height_score'], result['ethnicity_score'], result['education_score'],
                            result['occupation_score'], result['con_personality_score'], result['con_leadership_score'], result['con_hobby_score'], result['con_gender_score'], result['con_age_score'], result['con_height_score'], result['con_ethnicity_score'], result['con_education_score'], result['con_occupation_score'])

                        mycursor.execute(sql, val)
                        mysql.connection.commit()

            return render_template('miniGrps.html', fullSet=fullSet[0], grpAmt=grpAmt, biography=biography)
    return render_template('members.html', sid=sid, form=form, fullSet=fullSet, getMembers=getMembers, biography=biography)
    #         criteria = form.grpBy.data    # Compatible or Incompatible
    #         # Number of Persons in Each Group
    #         numPersons = int(form.numPersons.data)

    #         # Total Amount of Persons in Set
    #         length = len(getMembers)
    #         # How many Groups Formed
    #         grpAmt = round(len(getMembers)/numPersons)

    #         # mycursor.execute(
    #         #     'DELETE from SetUserGroup WHERE sid = %s', (sid,))

    #         # mycursor.execute(
    #         #     'DELETE from SetGroupScore WHERE sid = %s', (sid,))

    #         # # # CSI MAGIC

    #         # for i in range(grpAmt):
    #         #     sql = "INSERT INTO SetUserGroup (username, sid, group_num, leader) VALUES (%s, %s, %s, %s)"
    #         #     val = (results[i][0]['userA username '], sid, i+1, 1)

    #         #     mycursor.execute(sql, val)
    #         #     mysql.connection.commit()
    #         #     for result in results[i]:
    #         #         sql = "INSERT INTO SetUserGroup (username, sid, group_num, leader) VALUES (%s, %s, %s, %s)"
    #         #         val = (result['userB username'], sid, i+1, 0)

    #         #         mycursor.execute(sql, val)
    #         #         mysql.connection.commit()

    #         #         sql = "INSERT INTO SetGroupScore (`userA username`, `userB username`, sid, group_num, CSI, percentage, personality_score, leadership_score, hobby_score, gender_score, age_score, height_score, ethnicity_score, education_score, occupation_score, con_personality_score, con_leadership_score, con_hobby_score, con_gender_score, con_age_score, con_height_score, con_ethnicity_score, con_education_score, con_occupation_score) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

    #         #         val = (result['userA username '], result['userB username'], sid, i+1, result['CSI'],  result['Percentage'], result['personality_score'], result['leadership_score'], result['hobby_score'], result['gender_score'], result['age_score'], result['height_score'], result['ethnicity_score'], result['education_score'],
    #         #                result['occupation_score'], result['con_personality_score'], result['con_leadership_score'], result['con_hobby_score'], result['con_gender_score'], result['con_age_score'], result['con_height_score'], result['con_ethnicity_score'], result['con_education_score'], result['con_occupation_score'])

    #         #         mycursor.execute(sql, val)
    #         #         mysql.connection.commit()

    #         return render_template('miniGrps.html', fullSet=fullSet[0], grpAmt=grpAmt, biography=biography)
    # return render_template('members.html', sid=sid, form=form, fullSet=fullSet, getMembers=getMembers, biography=biography)


@app.route('/disband/<sid>')
def disband(sid):
    mycursor = mysql.connection.cursor()
    mycursor.execute('DELETE from SetUserGroup WHERE sid = %s', (sid,))
    mycursor.execute('DELETE from SetGroupScore WHERE sid = %s', (sid,))
    mysql.connection.commit()
    flash('Group disbanded', 'success')
    return redirect(url_for("members", sid=sid))


@app.route('/Group/<sid>', methods=["GET", "POST"])
def groupMembers(sid):
    numb = GroupNum()
    transfer = TranferGrp()

    mycursor = mysql.connection.cursor()
    mycursor.execute('SELECT * from sets WHERE sid = %s ', (sid,))
    fullSet = mycursor.fetchall()

    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()

    if request.method == "POST" and numb.validate_on_submit():
        group_num = int(numb.group_num.data)

        mycursor.execute(
            'SELECT first_name, last_name, user.username, leader from User JOIN SetUserGroup on SetUserGroup.username = user.username WHERE group_num = %s and sid=%s', (group_num, sid,))
        mems = mycursor.fetchall()

        mycursor.execute(
            'SELECT sum(CSI) as CSI, sum(percentage) as percentage, sum(personality_score) as personality_score, sum(leadership_score) as leadership_score, sum(hobby_score) as hobby_score, sum(gender_score) as gender_score, sum(age_score) as age_score, sum(height_score) as height_score, sum(ethnicity_score) as ethnicity_score, sum(education_score) as education_score, sum(occupation_score) as occupation_score, count(CSI) as amount from SetUserGroup JOIN SetGroupScore ON SetGroupScore.`userA username` = SetUserGroup.username AND SetGroupScore.group_num = SetUserGroup.group_num WHERE SetUserGroup.sid = %s AND SetGroupScore.group_num = %s', (sid, group_num, ))
        cur_set = mycursor.fetchall()
        print('>>>>>>>>>>>>>>>>>>>>>>>>', cur_set)
        return render_template('groupMembers.html', group_num=group_num, mems=mems, cur_set=cur_set[0], fullSet=fullSet[0], numb=numb, biography=biography, transfer=transfer)

    if request.method == "POST" and transfer.validate_on_submit():
        first_name = transfer.first_name.data
        last_name = transfer.last_name.data
        group_number = int(transfer.group_number.data)

        mycursor.execute(
            'SELECT first_name, last_name, user.username, group_num, leader from User JOIN SetUserGroup on SetUserGroup.username = user.username WHERE first_name = %s AND last_name = %s', (first_name, last_name, ))
        member = list(mycursor.fetchall())
 
        mycursor.execute(
            'SELECT DISTINCT count(group_num) as number from setusergroup WHERE group_num = %s', (int(member[0]['group_num']), ))
        curr_group = list(mycursor.fetchall())

        
        if member:
            if member[0]['leader'] == 0:
                if curr_group[0]['number'] > 2:
                    sql = "UPDATE SetUserGroup SET group_num = %s WHERE username = %s AND sid = %s"
                    val = (group_number, member[0]['username'], sid, )

                    mycursor.execute(sql, val)
                    mysql.connection.commit()

                    sql = "UPDATE SetGroupScore SET group_num = %s WHERE `userB username` = %s AND sid = %s"
                    val = (group_number, member[0]['username'], sid, )

                    mycursor.execute(sql, val)
                    mysql.connection.commit()
                    flash('Member has been tranferred', category="success")
                else:
                    flash(
                        'Member cannot be moved. At least 2 persons per group.', category="danger")
            else:
                flash('Cannot transfer group leaders to another group',
                      category="danger")
        else:
            flash('Member does not exist - Check Spelling', category="danger")

    mycursor.execute(
        'SELECT * from SetUserGroup where sid = %s ', (sid, ))
    groupsFormed = list(mycursor.fetchall())
    if groupsFormed:
        mycursor.execute(
            'SELECT max(group_num), count(group_num) from SetUserGroup where sid = %s ', (sid, ))
        grpAmt = list(mycursor.fetchall())
        return render_template('groupMembers.html', fullSet=fullSet[0], numb=numb, biography=biography, transfer=transfer)

    else:
        return redirect(url_for('members', sid=sid))


@app.route('/about/<typeUser>', methods=["GET", "POST"])
def aboutUser(typeUser):
    form = AboutYou()
    mycursor = mysql.connection.cursor()
    

    if request.method == "POST" and form.validate_on_submit():

        sql = """UPDATE Regular SET sex = %s, age = %s, height = %s, leadership = %s, ethnicity = %s, personality = %s, education = %s, hobby = %s, occupation = %s, pref_sex = %s, pref_ethnicity = %s WHERE user_id = %s"""
        val = (request.form['sex'], request.form['age'],
               request.form['height'], request.form['leadership'], request.form['ethnicity'], request.form['personality'], request.form['education'], request.form['hobby'], request.form['occupation'], request.form['pref_sex'], request.form['pref_ethnicity'], session.get('id'))
        mycursor.execute(sql, val)
        mysql.connection.commit()

        # Success Message Appears
        flash('Your information has been updated', 'success')

        # Redirect User to Main Page
        return redirect(url_for("dashboard", username=session.get('username')))

        # if form entry is invalid, redirected to the same page to fill in required details

    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()

    return render_template('about_you.html', form=form, biography=biography)


@app.route('/livesearch/', methods=["GET", "POST"])
def livesearch():
    searchbox = request.form.get("text")
    print('searchbox>>>>', searchbox)
    cursor = mysql.connection.cursor()
    
    
    
    cursor.execute( "select `userB username` from scores where `userB username` LIKE '%%{}%%' and `userA username`=%s  order by `userB username`".format(searchbox), (session.get('username'),))
    search_result = list(cursor.fetchall())
    if search_result!=[]:
        search_result= search_result
    
    print('serach result>>>>>>>>>', search_result)
    return jsonify(search_result)

@app.route('/aboutFriend/', methods=["GET", "POST"])
def aboutFriend():
    """Seek Compatibility between friend"""
    form = AboutFriend()
    mycursor = mysql.connection.cursor()

    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()

    if request.method == "POST" and form.validate_on_submit():
        username = request.form['username']

        # Check of username exists in database
        mycursor.execute(
            'SELECT * from user WHERE user.username = %s', (username,))

        userB = mycursor.fetchone()    # Friend

        print("userb data>>>>>>>>>>>>>>", userB)

        if userB:
           
            mycursor.execute(
                'SELECT * from scores WHERE `userA username` = %s AND `userB username` = %s', (session['username'], username, ))
            result = list(mycursor.fetchall())
            # Redirect User to Main Page
            print('>>>>>>>>>>>>>>>>> result', result)
            return render_template('friend_page.html', result=result[0], biography=biography)

        # if form entry is invalid, redirected to the same page to fill in required details
        flash("User does not exist. Try again.", category="danger")
    return render_template('about_friend.html', form=form, biography=biography)



# NEW FUNCTION !!!
# REDIRECTS TO MATCHES' PROFILE
@app.route('/profile/<username>', methods=["GET", "POST"])
def frndProfile(username):
    """Render the another user's dashboard page."""
    mycursor = mysql.connection.cursor()
    mycursor.execute(
        'SELECT * FROM user JOIN regular ON user.user_id = regular.user_id WHERE user.username = %s', (username,))
    user = mycursor.fetchone()

    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (user['user_id'],))
    frnbiography = mycursor.fetchone()

    # just so that the current user's image remains in the header
    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()

    # NEW TEMPLATE FRIENDPROFILE.HTML
    return render_template('friendprofile.html', biography=biography, frnbiography=frnbiography, user=user, friend=1)


# NEW FUNCTION !!!
# REDIRECTS TO USER FEATURE PROFILE
@app.route('/features/<username>', methods=["GET", "POST"])
def featureProfile(username):
    """Render the another user's dashboard page."""
    mycursor = mysql.connection.cursor()
    mycursor.execute(
        'SELECT * FROM user JOIN regular ON user.user_id = regular.user_id WHERE user.username = %s', (username,))
    user = mycursor.fetchone()

    # just so that the current user's image remains in the header
    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()

    # NEW TEMPLATE USER_FEATURES.HTML
    return render_template('user_features.html', biography=biography, user=user)




#  NEW FUNCTION !!!
#  FOR 'EX OFF' FUNCTIONALITY
@app.route('/block/<userB>', methods=['GET'])
def block(userB):
    """Accepts a username and 'EX OFF' that person from appearing in recommended matches """
    mycursor = mysql.connection.cursor()

    # EX OFF A PERSON FROM APPEARING IN RECOMMENDATIONS BY SETTING BLOCKED TO 1 (TRUE)
    sql = "UPDATE Scores SET blocked = 1  WHERE `userA username` = %s AND `userB username` = %s"
    val = (session['username'], userB)
    mycursor.execute(sql, val)
    mysql.connection.commit()

    return redirect(url_for('recommend', username=session.get('username')))








@app.route('/recommend/<username>', methods=['GET', 'POST'])
def recommend(username):
    """Render the website's recommended matches page."""
    form = Criteria()

    mycursor = mysql.connection.cursor()
    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()
    if request.method == "POST" and 'logged_in' in session:
        crit = form.crit.data
        if crit == "compatible":
            mycursor.execute(
                'SELECT * from Scores JOIN user on user.username=scores.`userB username` WHERE `userA username` = %s order by csi DESC LIMIT 9', (session['username'],))
        else:
            mycursor.execute('SELECT * from Scores JOIN user on user.username=scores.`userB username` WHERE `userA username` = %s order by csi ASC LIMIT 9', (session['username'],))
        matches = list(mycursor.fetchall())
        return render_template('recomnd.html', form=form, matches=matches, biography=biography)

    # mycursor.execute('SELECT * from Scores WHERE `userA username` = %s ORDER BY percentage DESC LIMIT 9', (session['username'],))
    mycursor.execute('SELECT * from Scores JOIN user on user.username=scores.`userB username` WHERE `userA username` = %s AND blocked = 0 AND `userA username` <> `userB username` order by csi DESC LIMIT 9', (session['username'],))
    matches = list(mycursor.fetchmany(9))
    

    return render_template('recomnd.html', form=form, matches=matches, biography=biography)


@app.route('/recommended/<match>', methods=['GET'])
def recommended(match):
    """Render the website's recommended matches page."""
    print("yolyo")
    mycursor = mysql.connection.cursor()
    mycursor.execute(

        'SELECT * from User JOIN Scores ON scores.`userA username`=user.username WHERE scores.`userA username` = %s AND `userB username` = %s', (session['username'], match, ))
    matched = mycursor.fetchone()

    print('>>>>>>>>>>>>>>>>>>',matched)
    mycursor.execute(
        'Select * from Biography WHERE user_id = %s', (session['id'],))
    biography = mycursor.fetchone()

    return render_template('match.html', match=matched, biography=biography)


# @app.route('/run')
# def run():
#     # CSI Magic
#     # comp_list is the variable for the list returned by CSI
#     # how many persons are we submitting to the database 25 ???
#     mycursor = mysql.connection.cursor()
#     mycursor.execute(
#         'DELETE from Scores WHERE `userA username` = %s', (session['username'],))

#     # Index 0 has in the 25 or so people to write to the db
#     for user in comp_list[0]:
#         # insert response to database
#         sql = "INSERT INTO Scores (`userA username`, `userB username`, CSI, percentage, personality_score, leadership_score, hobby_score, gender_score, age_score, height_score, ethnicity_score, education_score, occupation_score, con_personality_score, con_leadership_score, con_hobby_score, con_gender_score, con_age_score, con_height_score, con_ethnicity_score, con_education_score, con_occupation_score) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

#         # userA username has an extra space
#         val = (user['userA username '], user['userB username'], user['CSI'], int((user['CSI'] / 9) * 100), user['personality_score'], user['leadership_score'], user['hobby_score'], user['gender_score'], user['age_score'], user['height_score'], user['ethnicity_score'], user['education_score'],
#                user['occupation_score'], user['con_personality_score'], user['con_leadership_score'], user['con_hobby_score'], user['con_gender_score'], user['con_age_score'], user['con_height_score'], user['con_ethnicity_score'], user['con_education_score'], user['con_occupation_score'])

#         mycursor.execute(sql, val)
#         mysql.connection.commit()
#     return redirect(url_for('recommend', username=session.get('username')))


@app.route('/run')
def run():

    mycursor = mysql.connection.cursor()
    mycursor.execute("select count(*) as count from scores where `userA username`=%s",(session['username'],))
    num=mycursor.fetchone()
    num=num['count']

    if num==0:

        mycursor = mysql.connection.cursor()
        mycursor.execute(
            'SELECT username, first_name, last_name, sex, pref_sex, age, height, leadership, education, ethnicity, pref_ethnicity, hobby, occupation, personality from user join regular on user.user_id=regular.user_id WHERE user.username = %s', (session['username'],))
        cur_user = list(mycursor.fetchall())    # Current User
        # print("curr user:   ->>>>>>>>> ", cur_user)
        mycursor.execute(
            'SELECT username, first_name, last_name, sex, pref_sex, age, height, leadership, education, ethnicity, pref_ethnicity, hobby, occupation, personality from user join regular on user.user_id=regular.user_id WHERE NOT user.username = %s', (session['username'],))
        other_users = list(mycursor.fetchall())     # All but current user


        results,b9,t9= CSI_code.REGCSI  (cur_user,other_users)


        # Index 0 has in the 25 or so people to write to the db
        for user in results:
            # insert response to database
            sql = "INSERT INTO Scores (`userA username`, `userB username`,first_name,last_name, CSI, percentage, personality_score, leadership_score, hobby_score, gender_score, age_score, height_score, ethnicity_score, education_score, occupation_score, con_personality_score, con_leadership_score, con_hobby_score, con_gender_score, con_age_score, con_height_score, con_ethnicity_score, con_education_score, con_occupation_score) VALUES (%s, %s, %s, %s,%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

            # userA username has an extra space
            val = (user['userA username'], user['userB username'], user['firstname'],user['lastname'] ,user['CSI'], user['Percentage'] , user['personality_score'], user['leadership_score'], user['hobby_score'], user['gender_score'], user['age_score'], user['height_score'], user['ethnicity_score'], user['education_score'],
                user['occupation_score'], user['con_personality_score'], user['con_leadership_score'], user['con_hobby_score'], user['con_gender_score'], user['con_age_score'], user['con_height_score'], user['con_ethnicity_score'], user['con_education_score'], user['con_occupation_score'])

            mycursor.execute(sql, val)
            mysql.connection.commit()
    if num>0:
        mycursor = mysql.connection.cursor()
        mycursor.execute(
            'DELETE from Scores WHERE `userA username` = %s', (session['username'],))

        mycursor = mysql.connection.cursor()
        mycursor.execute(
            'SELECT username, first_name, last_name, sex, pref_sex, age, height, leadership, education, ethnicity, pref_ethnicity, hobby, occupation, personality from user join regular on user.user_id=regular.user_id WHERE user.username = %s', (session['username'],))
        cur_user = list(mycursor.fetchall())    # Current User
        # print("curr user:   ->>>>>>>>> ", cur_user)
        mycursor.execute(
            'SELECT username, first_name, last_name, sex, pref_sex, age, height, leadership, education, ethnicity, pref_ethnicity, hobby, occupation, personality from user join regular on user.user_id=regular.user_id WHERE NOT user.username = %s', (session['username'],))
        other_users = list(mycursor.fetchall())     # All but current user


        results,b9,t9= CSI_code.REGCSI  (cur_user,other_users)


        # Index 0 has in the 25 or so people to write to the db
        for user in results:
            # insert response to database
            sql = "INSERT INTO Scores (`userA username`, `userB username`,first_name,last_name, CSI, percentage, personality_score, leadership_score, hobby_score, gender_score, age_score, height_score, ethnicity_score, education_score, occupation_score, con_personality_score, con_leadership_score, con_hobby_score, con_gender_score, con_age_score, con_height_score, con_ethnicity_score, con_education_score, con_occupation_score) VALUES (%s, %s, %s, %s,%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

            # userA username has an extra space
            val = (user['userA username'], user['userB username'], user['firstname'],user['lastname'] ,user['CSI'], user['Percentage'] , user['personality_score'], user['leadership_score'], user['hobby_score'], user['gender_score'], user['age_score'], user['height_score'], user['ethnicity_score'], user['education_score'],
                user['occupation_score'], user['con_personality_score'], user['con_leadership_score'], user['con_hobby_score'], user['con_gender_score'], user['con_age_score'], user['con_height_score'], user['con_ethnicity_score'], user['con_education_score'], user['con_occupation_score'])

            mycursor.execute(sql, val)
            mysql.connection.commit()
    return redirect(url_for('recommend', username=session.get('username')))






@app.route('/users')
def getUsers():
    """Render website's home page."""
    mycursor = mysql.connection.cursor()

    mycursor.execute(
        'SELECT username, first_name, last_name, sex, pref_sex, age, height, leadership, education, ethnicity, pref_ethnicity, hobby, occupation, personality from user join regular on user.user_id=regular.user_id WHERE user.username = %s', (session['username'],))
    userA = list(mycursor.fetchall())    # Current User

    mycursor.execute(
        'SELECT username, first_name, last_name, sex, pref_sex, age, height, leadership, education, ethnicity, pref_ethnicity, hobby, occupation, personality from user join regular on user.user_id=regular.user_id WHERE NOT user.username = %s', (session['username'],))
    other_users = list(mycursor.fetchall())     # All but current user

    return '<p>' + str(cur_user) + '</p>' + '<p>' + str(other_users) + '</p>'


def randomFeatures():
    # These are random features for the regular user
    sex = random.choice(
        ['Female', 'Male'])
    pref_sex = random.choice(
        ['Female', 'Male'])
    height = random.randint(142, 198)
    age = random.randint(22, 35)
    leadership = random.choice(
        ['Autocratic', 'Laissez-Faire', 'Democratic'])
    hobby = random.choice(
        ['Sports', 'Music', 'Exercising', 'Shopping', 'Dancing', 'Watching-TV', 'Reading', 'Writing', 'Arts'])
    ethnicity = random.choice(
        ['Black', 'White', 'Chinese', 'Indian', 'Hispanic'])
    pref_ethnicity = random.choice(
        ['Black', 'White', 'Chinese', 'Indian', 'Hispanic'])
    occupation = random.choice(
        ['Business', 'Science', 'Technology', 'Construction', 'Communication', 'Law'])
    education = random.choice(
        ['Bachelors', 'Masters', 'PhD', 'Diploma', 'Associate Degree'])
    personality = random.choice(['Introvert', 'Extrovert', 'Ambivert'])

    return [sex, age, height, leadership, ethnicity, personality, education, hobby, occupation, pref_sex, pref_ethnicity]


@app.route('/results',  methods=['GET', 'POST'])
def result():
    """Render the website's result page."""
    return render_template('results.html')


# user_loader callback. This callback is used to reload the user object from
# the user ID stored in the session
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@login_manager.unauthorized_handler
def unauthorized():
    """Redirect unauthorized users to Login page."""
    flash('You must be logged in to view that page.')
    return redirect(url_for('login'))
###
# The functions below should be applicable to all Flask apps.
###


@app.route('/<file_name>.txt')
def send_text_file(file_name):
    """Send your static text file."""
    file_dot_text = file_name + '.txt'
    return app.send_static_file(file_dot_text)

# Flash errors from the form if validation fails


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"Error in the %s field - %s" % (
                getattr(form, field).label.text,
                error
            ), 'danger')


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.errorhandler(404)
def page_not_found(error):
    """Custom 404 page."""
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="8080")





# C70FC481C8

# AABE4AC2F6

# 56A90BA51F