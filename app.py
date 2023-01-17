from enum import unique
from flask import Flask
from flask import render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required
from flask_bootstrap import BOOTSTRAP_VERSION, Bootstrap
import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation
# from keras.layers import Activation
# from keras.optimizers import adam_v2
# from keras import metrics
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
# import pandas as pd
# import numpy as np
# import pandas_datareader.data as web
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt


from werkzeug.security import generate_password_hash, check_password_hash
import os

from datetime import datetime
import pytz
df = pd.read_csv("static\csv\output.csv")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SECRET_KEY'] = os.urandom(24)
db = SQLAlchemy(app)
bootstrap = Bootstrap(app)


login_manager = LoginManager()
login_manager.init_app(app)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(50), nullable=False)
    body = db.Column(db.String(300), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(pytz.timezone('Asia/Tokyo')))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), unique=True)
    password = db.Column(db.String(12))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'GET':
        posts = Post.query.all()
        return render_template('index.html', posts=posts)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User(username=username, password=generate_password_hash(password, method='sha256'))

        db.session.add(user)
        db.session.commit()
        return redirect('/login')
    else:
        return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if check_password_hash(user.password, password):
            login_user(user)
            return redirect('/')
    else:
        return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')


@app.route('/create', methods=['GET', 'POST'])
@login_required
def create():
    if request.method == 'POST':
        title = request.form.get('title')
        body = request.form.get('body')

        post = Post(title=title, body=body)

        db.session.add(post)
        db.session.commit()
        return redirect('/')
    else:
        return render_template('create.html')

@app.route('/<int:id>/update', methods=['GET', 'POST'])
@login_required
def update(id):
    post = Post.query.get(id)
    if request.method == 'GET':
        return render_template('update.html', post=post)
    else:
        post.title = request.form.get('title')
        post.body = request.form.get('body')

        db.session.commit()
        return redirect('/')

@app.route('/<int:id>/delete', methods=['GET'])
@login_required
def delete(id):
    post = Post.query.get(id)

    db.session.delete(post)
    db.session.commit()
    return redirect('/')

@app.route('/total')
# @login_required
def total():
    return render_template('total.html')

@app.route('/my_stock')
# @login_required
def my_stock():
    return render_template('my_stock.html')

@app.route('/favorite')
# @login_required
def favorite():
    return render_template('favorite.html')

@app.route('/company/<start>/<last>')
# @login_required
def company(start,last):
    return render_template('company.html',data_lists=df.values.tolist(),start=int(start),last=int(last))

@app.route('/predict')
# @login_required
def predict():
    c = {
    'chart_labels': "'選択肢１', '選択肢２', '選択肢３', '選択肢４','選択肢５'",
    'chart_data': "8, 5, 5, 5, 6",
    'chart_title': "レーダーサンプル",
    'chart_target': "Ａさん"
    }
    return render_template('predict.html', c = c)

#     c = {
#     nikkei = web.DataReader("NIKKEI225", "fred", "2019/9/27","2022/9/27") :
#     nikkei
#     date = range(len(nikkei["NIKKEI225"]))
#     value = nikkei["NIKKEI225"]
#     plt.plot(date, value)
#     nikkei = nikkei.interpolate()
#     data = nikkei.values
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     NikkeiData_norm = scaler.fit_transform(data)
#     NikkeiData_norm
#     maxlen = 10
#     x_data = []
#     y_data_price = []
#     for i in range(len(NikkeiData_norm) - maxlen): 
#     x_data.append(NikkeiData_norm[i:i + maxlen])      
#     y_data_price.append(NikkeiData_norm[i + maxlen]) 
#     x_data = np.asarray(x_data)
#     y_data_price = np.asarray(y_data_price)
#     train_size = int(x_data.shape[0] * 0.8)
#     x_train = x_data[:train_size] 
#     y_train_price = y_data_price[:train_size] 
#     x_test = x_data[train_size:]
#     y_test_price = y_data_price[train_size:]
#     out_neurons = 1
#     units = 300 
#     model = Sequential()
#     model.add(LSTM(units, batch_input_shape=(None, maxlen, 1), return_sequences=False))
#     model.add(Dense(out_neurons))
#     optimizer = adam_v2.Adam(lr=0.001)
#     model.compile(loss="mean_squared_error", optimizer=optimizer , metrics=[metrics.mae])
#     early_stopping = EarlyStopping(monitor='val_mean_absolute_error', mode='auto', patience=7)
#     hist = model.fit(x_train, y_train_price,
#             batch_size=30,
#             epochs=50,
#             validation_split=0.1,
#             callbacks=[early_stopping]
#             )
#     loss = hist.history['loss']
#     val_loss = hist.history['val_loss']
#     epochs = len(loss)
#     plt.plot(range(epochs), loss, label='loss(training data)')
#     plt.plot(range(epochs), val_loss, label='val_loss(evaluation data)')
#     plt.legend(loc='best')
#     plt.grid()
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.show()
#     predicted = model.predict(x_test)
#     predicted_N =    scaler.inverse_transform(predicted)
#     y_test_price_N = scaler.inverse_transform(y_test_price)
#     plt.plot(range(len(predicted)), predicted_N, marker='.', label='predicted')
#     plt.plot(range(len(y_test_price)), y_test_price_N, marker='.', label='y_test_price')
#     plt.grid()
#     plt.xlabel('DATE')
#     plt.ylabel('N225')
#     plt.show()
#     from sklearn.metrics import r2_score
#     r2_score(predicted_N,y_test_price_N)
# }
#     return render_template('templates\predict.html', c = c)


if __name__ == "__main__":
    app.run(debug=True)