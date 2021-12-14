import numpy as np, pandas as pd, tensorflow as tf
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

date_column = ['INTAKE TERM CODE', 'ADMIT TERM CODE', 'EXPECTED GRAD TERM CODE']
drop_column = ['ID 2', 'RECORD COUNT', 'PROGRAM LONG NAME', 'STUDENT TYPE NAME',
               'STUDENT TYPE GROUP NAME', 'PREV EDU CRED LEVEL NAME',
               'HS AVERAGE GRADE', 'PROGRAM SEMESTERS', 'TOTAL PROGRAM SEMESTERS',
               'RESIDENCY STATUS NAME', 'CURRENT STAY STATUS', 'APPL FIRST LANGUAGE DESC',
               'MAILING COUNTRY NAME', 'MAILING PROVINCE NAME', 'MAILING CITY NAME', 'MAILING POSTAL CODE']
data = pd.read_excel('HYPE dataset.xlsx', header=0).drop(columns=drop_column)
data['APPL EDUC INST TYPE NAME'] = data['APPL EDUC INST TYPE NAME'].fillna(0).replace('High School', 1)  # high school indicator
data.rename(columns={'APPL EDUC INST TYPE NAME': 'high school indicator', 'SUCCESS LEVEL': 'failure', 'APPLICANT CATEGORY NAME': 'effective academic history'}, inplace=True)
# column 'effective academic history' indicates history within certain years, it's ternary: no, high school, and post secondary
data['effective academic history'].replace({'Mature: Domestic 19 or older No Academic History': 'no', 'High School, Domestic': 'high school', 'BScN, High School Domestic': 'high school'}, inplace=True)
data['effective academic history'].replace(['Mature: Domestic  With Post Secondary', 'International Student, with Post Secondary'], 'post secondary', inplace=True)

# ****************************************************************************
# these columns contain illegal values, fill them with nan
data['GENDER'].replace('N', np.nan, inplace=True)
data['ACADEMIC PERFORMANCE'].replace('ZZ - Unknown', np.nan, inplace=True)
data['APPLICANT TARGET SEGMENT NAME'].replace('Unknown', np.nan, inplace=True)
# ****************************************************************************
international_postal = ['390', '682', '400', '143', '010']  # overseas zip codes
data['MAILING POSTAL CODE GROUP 3'].replace(international_postal, 'overseas', inplace=True)
data['failure'].replace(['In Progress', 'Successful', 'Unsuccessful'], [0, 0, 1], inplace=True)  # take 'in progress' and 'successful' as not failed
data['HS AVERAGE MARKS'][data['high school indicator'] == 0] = 0  # no mark for those who didn't attend high school
encoder = preprocessing.LabelEncoder()
for column in data.columns:
	data[column] = pd.Series(encoder.fit_transform(data[column][data[column].notna()]), index=data[column][data[column].notna()].index)
# impute missing values
data[['MAILING POSTAL CODE GROUP 3', 'FIRST GENERATION IND']] = KNNImputer().fit_transform(data[['MAILING POSTAL CODE GROUP 3', 'FIRST GENERATION IND']])
data[['APPLICANT TARGET SEGMENT NAME', 'MAILING POSTAL CODE GROUP 3', 'AGE GROUP LONG NAME']] = KNNImputer().fit_transform(data[['APPLICANT TARGET SEGMENT NAME', 'MAILING POSTAL CODE GROUP 3', 'AGE GROUP LONG NAME']])
data[['MAILING POSTAL CODE GROUP 3', 'ENGLISH TEST SCORE']] = KNNImputer().fit_transform(data[['MAILING POSTAL CODE GROUP 3', 'ENGLISH TEST SCORE']])
data[['high school indicator', 'HS AVERAGE MARKS']] = KNNImputer().fit_transform(data[['high school indicator', 'HS AVERAGE MARKS']])
data[['failure', 'ACADEMIC PERFORMANCE']] = KNNImputer().fit_transform(data[['failure', 'ACADEMIC PERFORMANCE']])
data[['PRIMARY PROGRAM CODE', 'GENDER']] = KNNImputer().fit_transform(data[['PRIMARY PROGRAM CODE', 'GENDER']])
data['GENDER'] = np.where(data['GENDER'] > .5, 1, 0)
data['FIRST GENERATION IND'] = np.where(data['FIRST GENERATION IND'] > .5, 1, 0)

reduced_feature = IncrementalPCA(n_components=1).fit_transform(data[['FUNDING SOURCE NAME', 'TIME STATUS NAME']])  # FUNDING SOURCE NAME is highly correlated with TIME STATUS NAME
data.insert(6, 'time and fund', reduced_feature)
data.drop(columns=['FUNDING SOURCE NAME', 'TIME STATUS NAME', 'high school indicator'], inplace=True)

label = data['failure']
data.drop(columns='failure', inplace=True)
data = StandardScaler().fit_transform(data)
dataset = tf.data.Dataset.from_tensor_slices((data, label))
for x, y in dataset:
	print(x, y)