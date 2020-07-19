import string
import uuid
import random
from . import db
from app.models import db
from werkzeug.security import generate_password_hash


class User(db.Model):
    __tablename__ = 'user'
    user_id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(20))
    first_name = db.Column(db.String(30))
    last_name = db.Column(db.String(30))
    email = db.Column(db.String(50), unique=True)
    username = db.Column(db.String(30), unique=True)
    password = db.Column(db.String(255))

    __mapper_args__ = {
        'polymorphic_identity': 'user',
        'polymorphic_on': type
    }

    def __init__(self, type, first_name, last_name, email, username, password):
        self.type = type
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.username = username
        self.password = generate_password_hash(
            password, method='pbkdf2:sha256')

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        try:
            return unicode(self.user_id)  # python 2 support
        except NameError:
            return str(self.user_id)  # python 3 support

    def __repr__(self):
        return '<User %r>' % (self.username)


class Administrator (User):
    __tablename__ = 'administrator'

    user_id = db.Column(db.Integer, db.ForeignKey(
        'user.user_id'), primary_key=True)
    position = db.Column(db.String(30))
    groups = db.relationship('Sets', backref='admin')

    def __init__(self, type, first_name, last_name, email, username, password, position):
        super().__init__(type, first_name, last_name, email, username, password)
        self.position = position

    __mapper_args__ = {
        'polymorphic_identity': 'Administrator'
    }


class Regular (User):
    __tablename__ = 'regular'

    user_id = db.Column(db.Integer, db.ForeignKey(
        'user.user_id'), primary_key=True)
    gender = db.Column(db.String(50))
    age = db.Column(db.String(50))
    height = db.Column(db.String(50))
    leadership = db.Column(db.String(30))
    ethnicity = db.Column(db.String(30))
    personality = db.Column(db.String(30))
    education = db.Column(db.String(50))
    hobby = db.Column(db.String(50))
    faculty = db.Column(db.String(50))
    occupation = db.Column(db.String(50))

    def __init__(self, type, first_name, last_name, email, username, password, gender, age, height, leadership, ethnicity, personality, education, hobby, faculty, occupation):
        super().__init__(type, first_name, last_name, email, username, password)
        self.ethnicity = ethnicity
        self.age = age
        self.height = height
        self.personality = personality
        self.leadership = leadership
        self.gender = gender
        self.hobby = hobby
        self.education = education
        self.faculty = faculty
        self.occupation = occupation

    __mapper_args__ = {
        'polymorphic_identity': 'Regular'
    }


class Sets (db.Model):
    __tablename__ = 'Sets'

    sid = db.Column(db.Integer, primary_key=True)
    set_name = db.Column(db.String(20), unique=True)
    purpose = db.Column(db.String(30))
    code = db.Column(db.String(10))
    administrator = db.Column(db.Integer, db.ForeignKey(
        'administrator.user_id'))

    def __init__(self, set_name, purpose, administrator):
        def random_Coder(lgth):
            """Generate a random string of letters, digits and special characters """
            return ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation) for i in range(lgth))

        def random_Coder2(lgth):
            """Generate a random string of letters, digits and special characters Option 2"""
            return uuid.uuid4().hex.upper()[0:lgth]

        self.set_name = set_name
        self.purpose = purpose
        self.administrator = administrator
        # self.code = random_Coder(10)
        self.code = random_Coder2(10)

    def get_id(self):
        try:
            return unicode(self.sid)  # python 2 support
        except NameError:
            return str(self.sid)  # python 3 support

    def get_Code(self):
        try:
            return unicode(self.code)  # python 2 support
        except NameError:
            return str(self.code)  # python 3 support


class joinSet(db.Model):
    __tablename__ = 'joinSet'

    user_id = db.Column(db.Integer, db.ForeignKey(
        'regular.user_id'), primary_key=True)
    sid = db.Column(db.Integer, db.ForeignKey(
        'Sets.sid'), primary_key=True)

    def get_gid(self):
        try:
            return unicode(self.sid)  # python 2 support
        except NameError:
            return str(self.sid)  # python 3 support


class SetUserGp(db.Model):
    __tablename__ = 'SetUserGp'

    user_id = db.Column(db.Integer, db.ForeignKey(
        'regular.user_id'), primary_key=True)
    sid = db.Column(db.Integer, db.ForeignKey(
        'Sets.sid'), primary_key=True)
    gp_num = db.Column(db.Integer,  primary_key=True)

    def get_gid(self):
        try:
            return unicode(self.sid)  # python 2 support
        except NameError:
            return str(self.sid)  # python 3 support


class Scores(db.Model):
    __tablename__ = 'userScore'

    user_id = db.Column(db.Integer, db.ForeignKey(
        'regular.user_id'), primary_key=True)
    feature = db.Column(db.String(20), primary_key=True)
    weight = db.Column(db.DECIMAL(2, 1))
