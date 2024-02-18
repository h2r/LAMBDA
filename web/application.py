from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import os
from dynamodb import generate_unique_id, insert_unique_id
from s3 import save_csv_to_s3

application = Flask(__name__)
application.secret_key = 'af@93k$j392}a' 

@application.route('/')
def index():
    if 'user_id' not in session:
        unique_id = generate_unique_id()
        insert_unique_id(unique_id)
        session['user_id'] = unique_id
    else:
        unique_id = session['user_id']
    return render_template('index.html', user_id=unique_id)

@application.route('/scenes')
def scenes():
    return render_template('scenes.html')

@application.route('/end')
def end():
    return render_template('end.html')

@application.route('/submit', methods=['POST'])
def submit_data():
    scene_data = request.json

    if 'all_scene_data' not in session:
        session['all_scene_data'] = []

    session['all_scene_data'].append(scene_data)

    # Flask sessions are stored client-side in cookies by default,
    # so you must explicitly mark the session as modified to ensure it gets saved
    session.modified = True

    return jsonify({'message': 'Data received and stored in session.'})

@application.route('/save', methods=['POST'])
def save_data():
    unique_id = session['user_id']

    all_data = session.get('all_scene_data', [])

    csv_filename = f'commands_participant{unique_id}.csv'
    df = pd.DataFrame(all_data)
    csv_content = df.to_csv(csv_filename, index=False)

    save_csv_to_s3(csv_content, csv_filename)

    if os.path.exists(csv_filename):
        os.remove(csv_filename)
        print(f"{csv_filename} has been deleted.")

    session.pop('all_scene_data', None)

    return jsonify({'message': 'Data saved successfully'})


if __name__ == '__main__':
    application.run()