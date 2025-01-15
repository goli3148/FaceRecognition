from app import app, db
import os
from flask import request, redirect, url_for, jsonify
from .database.models import People

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/api/UnlabelFace', methods=['POST'])
def newFace():
    uploded_file = request.files['file']
    label = request.form['label']
    if uploded_file.filename != '':
        save_dir = os.path.join(app.config["UPLOAD_DIR"], label)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, uploded_file.filename)
        uploded_file.save(save_dir)
        if 'name' in request.form:
            new_ = People(label=label, image=save_dir, name=request.form['name'])
        else:
            new_ = People(label=label, image=save_dir)
        db.session.add(new_)
        db.session.commit()
        return redirect(url_for('index')), 200
    return redirect(url_for('index')), 500

@app.route('/api/NewLabel')
def getNewLabel():
    try:
        query = db.session.query(People).order_by(People.id.asc()).all()[-1]
        new_label = {'new label' : query.label + 1}
        return jsonify(new_label)
    except:
        return jsonify({'new label': 0})
