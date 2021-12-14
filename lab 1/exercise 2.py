import pandas as pd, numpy as np, pandas_profiling as profile
from sklearn.impute import KNNImputer
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

data = pd.read_csv('hepatitis').replace('?', np.nan)
data['class'].replace(2, 0, inplace=True)
report = profile.ProfileReport(data)
report.to_file('EDA')
# fill missing values
data[['steroid', 'malaise', 'age']] = KNNImputer().fit_transform(data[['steroid', 'malaise', 'age']])
data[['liver big', 'liver firm', 'alk phosphate', 'spiders', 'fatigue']] = KNNImputer().fit_transform(data[['liver big', 'liver firm', 'alk phosphate', 'spiders', 'fatigue']])
data[['spleen palpable', 'alk phosphate', 'spiders']] = KNNImputer().fit_transform(data[['spleen palpable', 'alk phosphate', 'spiders']])
data[['ascites', 'varices', 'albumin', 'fatigue']] = KNNImputer().fit_transform(data[['ascites', 'varices', 'albumin', 'fatigue']])
data[['bilirubin', 'alk phosphate', 'varices', 'albumin']] = KNNImputer().fit_transform(data[['bilirubin', 'alk phosphate', 'varices', 'albumin']])
data[['sgot', 'alk phosphate', 'anorexia']] = KNNImputer().fit_transform(data[['sgot', 'alk phosphate', 'anorexia']])
data[['protime', 'age']] = KNNImputer().fit_transform(data[['protime', 'age']])
data['steroid'] = np.where(data['steroid'] > 1.5, 2, 1)
data['liver big'] = np.where(data['liver big'] > 1.5, 2, 1)
data['liver firm'] = np.where(data['liver big'] > 1.5, 2, 1)
data['spiders'] = np.where(data['spiders'] > 1.5, 2, 1)
data['fatigue'] = np.where(data['fatigue'] > 1.5, 2, 1)
data['spleen palpable'] = np.where(data['spleen palpable'] > 1.5, 2, 1)
data['ascites'] = np.where(data['ascites'] > 1.5, 2, 1)
data['varices'] = np.where(data['varices'] > 1.5, 2, 1)
X, Y = data.loc[:, 'age':], data['class']
# balance samples
X, Y = ADASYN().fit_resample(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

# modeling
model = MLPClassifier(hidden_layer_sizes=(30, 15), activation='logistic', max_iter=1000000)
model.fit(x_train, y_train)
predict = model.predict(x_test)
print(classification_report(y_test, predict))