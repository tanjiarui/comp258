import pandas as pd, shap, catboost as cb
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report

data = pd.read_csv('data')
train_x, train_y = data[data['INTAKE TERM CODE'] < 2020].drop(columns='failure'), data[data['INTAKE TERM CODE'] < 2020]['failure']
test_x, test_y = data[data['INTAKE TERM CODE'] >= 2020].drop(columns='failure'), data[data['INTAKE TERM CODE'] >= 2020]['failure']
test_x, test_y = ADASYN().fit_resample(test_x, test_y)  # test set is unbalanced

# num_classes = 2
# num_trees = 26
# depth = 6
# used_features_rate = 0.5
#
# def create_forest_model():
# 	inputs = tf.keras.Input(shape=(train_x.shape[1],))
# 	features = tf.keras.layers.BatchNormalization()(inputs)
# 	num_features = features.shape[1]
# 	forest_model = NeuralDecisionForest(num_trees, depth, num_features, used_features_rate, num_classes)
# 	outputs = forest_model(features)
# 	model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 	return model
# forest_model = create_forest_model()
# forest_model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
# forest_model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100)

boost = cb.CatBoostClassifier()
boost.fit(train_x, train_y)
predict = boost.predict(test_x)
print(classification_report(test_y, predict))
boost.save_model('model')

explainer = shap.TreeExplainer(boost)
shap_values = explainer.shap_values(test_x)
shap.summary_plot(shap_values, test_x)