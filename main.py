import os

from flask import Flask, render_template
import flask_socketio as socketio


app = Flask(__name__)
app.secret_key = os.urandom(16)
socket = socketio.SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/phointer')
def phointer():
    return render_template('phointer.html')

@socket.on('join')
def handle_join(json, methods=['GET', 'POST']):
    socketio.join_room(json['as'])

@socket.on('orientation')
def handle_orientation(json, methods=['GET', 'POST']):
    print(str(json))
    socketio.emit('orientation', json, room='monitor')

@socket.on('record')
def handle_record(json, methods=['GET', 'POST']):
    print(str(json))
    socketio.emit('record', json, room='monitor')


if __name__ == '__main__':
    socket.run(app, debug=True, host='0.0.0.0', port=8888, certfile='cert.pem', keyfile='key.pem')
