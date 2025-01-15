from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, FileField
from wtforms.validators import DataRequired

class UploadUnlabelFace(FlaskForm):
    image = FileField('images', validators=[DataRequired()])
    submit = SubmitField('Upload')
