import os

from flask import Flask, render_template
import flask_socketio as socketio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


app = Flask(__name__)
app.secret_key = os.urandom(16)
socket = socketio.SocketIO(app)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data=[]):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def append(self, x):
        self.data.append(x)

dataset = Dataset()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x

net = Net().to(dtype=torch.float32, device=torch.device('cpu'))
print(net)


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
    net.eval()
    abc = [json['alpha'] / 360,
           json['beta'] / 360 + 0.5,
           json['gamma'] / 180 + 0.5]
    x, y = net(torch.tensor([abc], dtype=torch.float32, device=torch.device('cpu')))[0].tolist()
    json['estimated_x'] = x
    json['estimated_y'] = y

    socketio.emit('orientation', json, room='monitor')


@socket.on('record')
def handle_record(json, methods=['GET', 'POST']):
    print(str(json))
    socketio.emit('record', json, room='monitor')


@socket.on('train')
def handle_train(json, methods=['GET', 'POST']):
    source = json['source']
    target = json['target']

    source[0] = source[0] / 360
    source[1] = source[1] / 360 + 0.5
    source[2] = source[2] / 180 + 0.5

    dataset.append([source, target])
    print([source, target])
    print(len(dataset))

    if len(dataset) % 1000 == 0:
        print('Start training')
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=1e-1)

        train_dataset_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                drop_last=False,
                collate_fn=lambda x: x)
        net.train()
        for epoch in range(10000):
            epoch_loss = 0
            for i, batch in enumerate(train_dataset_loader):
                xs, ys = zip(*batch)

                # Convert into tensors
                xs = torch.tensor(xs, dtype=torch.float32, device=torch.device('cpu'))
                ys = torch.tensor(ys, dtype=torch.float32, device=torch.device('cpu'))

                # Update the model
                optimizer.zero_grad()
                output = net(xs)
                loss = criterion(output, ys)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.sum()
            print(f'Epoch: {epoch}, {epoch_loss / len(dataset)}')
        print('End training')


if __name__ == '__main__':
    socket.run(app, debug=True, host='0.0.0.0', port=8888, certfile='cert.pem', keyfile='key.pem')
