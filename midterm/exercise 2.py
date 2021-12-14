import pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import regularizers
from tensorflow.keras.optimizers import Adamax
from keras.callbacks import EarlyStopping

data = pd.read_csv('diabetes noheaders.csv', header=None)
x_train, x_test, y_train, y_test = train_test_split(data.loc[:, :7], data.loc[:, 8], test_size=.2)

def build_model(activation='relu', optimizer=Adamax()):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(32, activation=activation, kernel_regularizer=regularizers.l2()))
	model.add(tf.keras.layers.Dense(16, activation=activation, kernel_regularizer=regularizers.l2()))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

model = build_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, callbacks=[early_stopping])
print('test set')
print(x_test)
print('label')
print(y_test)
print('prediction')
print(model.predict(x_test))