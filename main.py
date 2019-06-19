import os

from flask import Flask, render_template
from flask_socketio import SocketIO


app = Flask(__name__)
app.secret_key = os.urandom(16)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('orientation')
def handle_orientation(json, methods=['GET', 'POST']):
    print(str(json))


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8888, certfile='cert.pem', keyfile='key.pem')
