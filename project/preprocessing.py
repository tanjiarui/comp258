import numpy as np, pandas as pd, ppscore as pps, matplotlib.pyplot as plt, seaborn as sns, pickle#, pandas_profiling as profile
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.decomposition import IncrementalPCA

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
for column in date_column:
	data[column] = pd.to_datetime(data[column])
# report = profile.ProfileReport(data)
# report.to_file('EDA')
for column in date_column:
	data[column] = pd.to_datetime(data[column]).dt.year

# ****************************************************************************
# these columns contain illegal values, fill them with nan
data['GENDER'].replace('N', np.nan, inplace=True)
data['ACADEMIC PERFORMANCE'].replace('ZZ - Unknown', np.nan, inplace=True)
data['APPLICANT TARGET SEGMENT NAME'].replace('Unknown', np.nan, inplace=True)
# ****************************************************************************
international_postal = ['390', '682', '400', '143', '010']  # overseas zip codes
data['MAILING POSTAL CODE GROUP 3'].replace(international_postal, 'overseas', inplace=True)
data['failure'].replace(['In Progress', 'Successful', 'Unsuccessful'], [0, 0, 1], inplace=True)  # take 'in progress' and 'successful' as not failed
data.loc[data[data['high school indicator'] == 0].index, 'HS AVERAGE MARKS'] = 0  # no mark for those who didn't attend high school
categorical_column = [col for col in data.columns if data[col].dtype == 'object'] + ['PRIMARY PROGRAM CODE', 'INTAKE TERM CODE', 'ADMIT TERM CODE', 'EXPECTED GRAD TERM CODE']
label_encoders = dict()
for column in categorical_column:
	encoder = preprocessing.LabelEncoder()
	data[column] = pd.Series(encoder.fit_transform(data[column][data[column].notna()]), index=data[column][data[column].notna()].index)
	label_encoders[column] = encoder
# impute missing values
data[['MAILING POSTAL CODE GROUP 3', 'FIRST GENERATION IND']] = KNNImputer().fit_transform(data[['MAILING POSTAL CODE GROUP 3', 'FIRST GENERATION IND']])
data[['APPLICANT TARGET SEGMENT NAME', 'MAILING POSTAL CODE GROUP 3', 'AGE GROUP LONG NAME']] = KNNImputer().fit_transform(data[['APPLICANT TARGET SEGMENT NAME', 'MAILING POSTAL CODE GROUP 3', 'AGE GROUP LONG NAME']])
data[['MAILING POSTAL CODE GROUP 3', 'ENGLISH TEST SCORE']] = KNNImputer().fit_transform(data[['MAILING POSTAL CODE GROUP 3', 'ENGLISH TEST SCORE']])
data[['high school indicator', 'HS AVERAGE MARKS']] = KNNImputer().fit_transform(data[['high school indicator', 'HS AVERAGE MARKS']])
data[['failure', 'ACADEMIC PERFORMANCE']] = KNNImputer().fit_transform(data[['failure', 'ACADEMIC PERFORMANCE']])
data[['PRIMARY PROGRAM CODE', 'GENDER']] = KNNImputer().fit_transform(data[['PRIMARY PROGRAM CODE', 'GENDER']])
data['effective academic history'] = np.where(data['high school indicator'] == 1, 0, 1)
data['GENDER'] = np.where(data['GENDER'] > .5, 1, 0)
data['FIRST GENERATION IND'] = np.where(data['FIRST GENERATION IND'] > .5, 1, 0)
data['MAILING POSTAL CODE GROUP 3'] = round(data['MAILING POSTAL CODE GROUP 3'])
data['ACADEMIC PERFORMANCE'] = round(data['ACADEMIC PERFORMANCE'])
data['ENGLISH TEST SCORE'] = round(data['ENGLISH TEST SCORE'])

pca = IncrementalPCA(n_components=1).fit(data[['FUNDING SOURCE NAME', 'TIME STATUS NAME']])
reduced_feature = pca.transform(data[['FUNDING SOURCE NAME', 'TIME STATUS NAME']])  # FUNDING SOURCE NAME is highly correlated with TIME STATUS NAME
data.insert(6, 'time and fund', reduced_feature)
data.drop(columns=['FUNDING SOURCE NAME', 'TIME STATUS NAME', 'high school indicator'], inplace=True)
data.to_csv('data', index=False)
file = open('encoder', 'wb')
serializer = {'label encoders': label_encoders, 'pca': pca}
pickle.dump(serializer, file)
file.close()

# non-linear correlations
matrix = pps.matrix(data)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
sns.heatmap(matrix, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
plt.show()