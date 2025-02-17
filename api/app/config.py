import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    UPLOAD_DIR = os.environ.get('UPLOAD_DIR') or 'api/static/images'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///'+os.path.join(basedir, 'database/app.db')