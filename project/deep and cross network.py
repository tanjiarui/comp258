import pandas as pd, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('data')
# stratified sampling
x_train, x_test, y_train, y_test = train_test_split(data.drop(columns='failure'), data['failure'], stratify=data['failure'], test_size=.2)

hidden_units = [64, 64]
def create_deep_and_cross_model():
	input_layer = tf.keras.Input(shape=x_train.shape[1])
	batch_norm = tf.keras.layers.BatchNormalization()(input_layer)

	cross = batch_norm
	for _ in hidden_units:
		units = batch_norm.shape[-1]
		x = tf.keras.layers.Dense(units, kernel_regularizer='l2')(batch_norm)
		cross = batch_norm * x + cross
	cross = tf.keras.layers.BatchNormalization()(cross)

	deep = batch_norm
	for units in hidden_units:
		deep = tf.keras.layers.Dense(units, kernel_regularizer='l2')(deep)
		deep = tf.keras.layers.BatchNormalization()(deep)
		deep = tf.keras.layers.ReLU()(deep)

	merged = tf.keras.layers.concatenate([cross, deep])
	output_layer = tf.keras.layers.Dense(units=2, activation='sigmoid')(merged)
	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	model.compile(loss='sparse_categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
	return model

model = create_deep_and_cross_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)
predict = model.predict(x_test)
predict = [np.argmax(one_hot) for one_hot in predict]
print(classification_report(y_test, predict))
model.save('deep and cross network')


'''
				precision   recall    f1-score    support
			0.0     0.89    0.90        0.89        79
			1.0     0.85    0.84        0.84        55
	accuracy                            0.87        134
	macro avg       0.87    0.87        0.87        134
weighted avg        0.87    0.87        0.87        134
'''
# X, Y = data.drop(columns='failure'), data['failure']
# kfold = StratifiedKFold(shuffle=True)
# scores = []
# for train, test in kfold.split(X, Y):
# 	model.fit(X.loc[train], Y[train], epochs=100)
# 	score = model.evaluate(X.loc[test], Y[test])
# 	print('%s: %.2f' % (model.metrics_names[1], score[1]))
# 	scores.append(score[1])
# print("%.2f (+/- %.2f)" % (np.mean(scores), np.std(scores)))  # 0.95 (+/- 0.04)