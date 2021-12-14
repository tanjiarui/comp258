import tensorflow as tf
from keras.datasets import cifar10
from keras import regularizers
from tensorflow.keras.optimizers import Adamax
from keras.callbacks import EarlyStopping, TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# one hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train /= 255
x_test /= 255

def build_model(activation='relu', optimizer=Adamax(decay=.001/100)):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Flatten(input_shape=[32, 32, 3]))
	model.add(tf.keras.layers.Dense(1000, activation=activation, kernel_regularizer=regularizers.l2()))
	model.add(tf.keras.layers.Dense(400, activation=activation, kernel_regularizer=regularizers.l2()))
	model.add(tf.keras.layers.Dense(300, activation=activation, kernel_regularizer=regularizers.l2()))
	model.add(tf.keras.layers.Dense(200, activation=activation, kernel_regularizer=regularizers.l2()))
	model.add(tf.keras.layers.Dense(100, activation=activation, kernel_regularizer=regularizers.l2()))
	model.add(tf.keras.layers.Dense(50, activation=activation, kernel_regularizer=regularizers.l2()))
	model.add(tf.keras.layers.Dense(10, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

# tuning hyper parameters
model = KerasClassifier(build_model)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
param_grid = {'activation': ['relu', 'tanh', 'linear'], 'optimizer': ['Adam', 'Adamax', 'Nadam']}
grid_search = GridSearchCV(model, param_grid, n_jobs=2)
grid_search.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, callbacks=[early_stopping])
print('best parameter:')
print(grid_search.best_params_)  # {'activation': 'relu', 'optimizer': 'Adamax'}
print('best score:')
print(grid_search.best_score_)  # 0.49444000124931337

# converge the loss
model = tf.keras.models.load_model('cifar10')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
tensorboard = TensorBoard()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, callbacks=[early_stopping, tensorboard])
model.save('cifar10')