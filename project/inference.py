import pandas as pd, pickle, tensorflow as tf

sample = [{'INTAKE TERM CODE': 2020, 'ADMIT TERM CODE': 2020, 'INTAKE COLLEGE EXPERIENCE': 'New to College', 'PRIMARY PROGRAM CODE': 9111, 'SCHOOL CODE': 'CH',
                'STUDENT LEVEL NAME': 'Post Secondary', 'MAILING POSTAL CODE GROUP 3': 'L1V', 'GENDER': 'F', 'DISABILITY IND': 'N', 'FUTURE TERM ENROL': '1-1-1-1-0-0-0-0-0-0',
                'ACADEMIC PERFORMANCE': 'DF - Poor', 'EXPECTED GRAD TERM CODE': 2020, 'FIRST YEAR PERSISTENCE COUNT': 0, 'HS AVERAGE MARKS': 0, 'ENGLISH TEST SCORE': 140,
				'AGE GROUP LONG NAME': '41 to 50', 'FIRST GENERATION IND': 'Y', 'effective academic history': 'no', 'APPLICANT TARGET SEGMENT NAME': 'Non-Direct Entry',
				'TIME STATUS NAME': 'Full-Time', 'FUNDING SOURCE NAME': 'GPOG - FT'}]
sample = pd.DataFrame(sample)
file = open('encoder', 'rb')
deserializer = pickle.load(file)
label_encoders = deserializer['label encoders']
pca = deserializer['pca']
file.close()

# encoding categorical features
categorical_column = [col for col in sample.columns if sample[col].dtype == 'object'] + ['PRIMARY PROGRAM CODE', 'INTAKE TERM CODE', 'ADMIT TERM CODE', 'EXPECTED GRAD TERM CODE']
for column in categorical_column:
	encoder = label_encoders[column]
	print(column + ':')
	print(encoder.classes_)
	sample[column] = pd.Series(encoder.transform(sample[column][sample[column].notna()]), index=sample[column][sample[column].notna()].index)
# feature reduction
reduced_feature = pca.transform(sample[['FUNDING SOURCE NAME', 'TIME STATUS NAME']])  # FUNDING SOURCE NAME is highly correlated with TIME STATUS NAME
sample.insert(6, 'time and fund', reduced_feature)
sample.drop(columns=['FUNDING SOURCE NAME', 'TIME STATUS NAME'], inplace=True)

model = tf.keras.models.load_model('deep and cross network')
prediction = model.predict(sample)[:, 1][0]
print(prediction)