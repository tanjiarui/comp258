import os, numpy as np, tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# load data
codes, lines = list(), 1000
for root, dirs, files in os.walk('fake tensorflow'):
	for file in files:
		if file.find('.') != -1 and file.split('.')[1] == 'py':
			file = open(os.path.join(root, file), 'r')
			for line in file:
				if len(line) > 4:
					codes.append(line)
					lines -= 1
				if lines == 0:
					break
			file.close()
		if lines == 0:
			break
	if lines == 0:
		break

codes = ''.join(codes)
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(codes)
n_steps, stride = 100, 1
X, Y = list(), list()
for i in range(0, len(codes) - n_steps, stride):
	X.append(tokenizer.texts_to_matrix(codes[i: i + n_steps], 'tfidf'))
	Y.append(tokenizer.texts_to_matrix(codes[i + stride: i + n_steps + stride], 'tfidf'))
X, Y = np.array(X), np.array(Y)
del codes

def build_model():
	input_layer = tf.keras.Input(shape=(n_steps, len(tokenizer.index_word) + 1))
	norm = tf.keras.layers.LayerNormalization()(input_layer)
	lstm = tf.keras.layers.LSTM(256, return_sequences=True, kernel_regularizer='l2')(norm)
	dense = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2')(lstm)
	norm = tf.keras.layers.LayerNormalization()(dense)
	attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=4, kernel_regularizer='l2')(norm, norm)
	output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(tokenizer.index_word) + 1, activation='softmax'))(attention)
	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
	return model

model = build_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
model.fit(X, Y, validation_split=.2, epochs=10, callbacks=[early_stopping])
model.save('model')

model = tf.keras.models.load_model('model')
query = '''
def enable():
  # Enables v2 behaviors.
  _pywrap_tf2.enable(True)

def disable():
  # Disables v2 behaviors.
  _pywrap_tf2.enable(False)

@tf_export("__internal__.tf2.enabled", v1=[])
def enabled():
  # Returns True iff TensorFlow 2.0 behavior should be enabled.
  return _pywrap_tf2.is_enabled()'''
query = tokenizer.texts_to_matrix(query[:100], 'tfidf')
predict = model.predict(query.reshape([-1, query.shape[0], query.shape[1]])).squeeze()
predict = [np.argmax(one_hot) for one_hot in predict]
print(tokenizer.sequences_to_texts([predict]))