from flask import Flask, request
# from flask_cors import cross_origin
import pandas as pd, pickle, tensorflow as tf

"""
LOAD MODEL
"""
file = open('encoder', 'rb')
deserializer = pickle.load(file)
label_encoders = deserializer['label encoders']
pca = deserializer['pca']
file.close()

model = tf.keras.models.load_model('deep and cross network')
"""
SERVER CONFIG
"""
app = Flask(__name__) # create app instance


"""
ROUTES
"""
# @app.route('/predict', methods=['POST'])
# @cross_origin()
# def predict():
# 	# 1) grab POST request data
# 	form = request.json
#
# 	# 2) define form keys as dataset columns for matching model input columns
# 	form_column_dict = {
# 	'intakeTermCode': 'INTAKE TERM CODE',
# 	'admitTermCode': 'ADMIT TERM CODE',
# 	'expectedGradTermCode': 'EXPECTED GRAD TERM CODE',
# 	'primaryProgramCode': 'PRIMARY PROGRAM CODE',
# 	'hsAverageMarks': 'HS AVERAGE MARKS',
# 	'englishTestScore': 'ENGLISH TEST SCORE',
# 	'firstYearPersistenceCount': 'FIRST YEAR PERSISTENCE COUNT',
# 	'intakeCollegeExperience': 'INTAKE COLLEGE EXPERIENCE',
# 	'schoolCode': 'SCHOOL CODE',
# 	'studentLevelName': 'STUDENT LEVEL NAME',
# 	'timeStatusName': 'TIME STATUS NAME',
# 	'fundingSourceName': 'FUNDING SOURCE NAME',
# 	'mailingPostalCodeGroup3': 'MAILING POSTAL CODE GROUP 3',
# 	'gender': 'GENDER',
# 	'disabilityInd': 'DISABILITY IND',
# 	'futureTermEnroll': 'FUTURE TERM ENROL',
# 	'academicPerformance': 'ACADEMIC PERFORMANCE',
# 	'ageGroupLongName': 'AGE GROUP LONG NAME',
# 	'firstGenerationInd': 'FIRST GENERATION IND',
# 	'effectiveAcademicHistory': 'EFFECTIVE ACADEMIC HISTORY',
# 	'applicantTargetSegmentName': 'APPLICANT TARGET SEGMENT NAME'
# 	}
#
# 	# 3) format data into model input dataframe format
# 	pred_sample = {}
# 	print("\nFORM KEYS, VALUES and TYPES")
# 	print("-----------------------------")
# 	for key, value in form.items():
# 		print(key, ":>>", value, ":>>", type(value))
# 		pred_sample[form_column_dict.get(key)] = value
# 		print("\n")
#
# 	# 4) final format
# 	pred_sample=[pred_sample]
# 	print("FORMATED FORM (ready to transform into a DF)")
# 	print("--------------------------------------------")
# 	print(type(pred_sample))
# 	print(pred_sample)
# 	print("\n")
#
#
# 	return form
	# prediction = model.predict(arr)
	# print("\nPrediction\n",prediction)
	# if prediction[0] == 0:
	#     return render_template('home.html', prediction_text='Model predicts that the bike should be STOLEN!')
	# elif prediction[0] == 1:
	#     return render_template('home.html', prediction_text='Model predicts that the bike should be RECOVERED!')


## Start Flash Server
# Since there is no main() function in Python,
# when the command to run a python program is given to the interpreter,
# the code that is at level 0 indentation is to be executed.
# However, before doing that, it will define a few special variables.
# __name__ is one such special variable. If the source file is executed as the main program,
# the interpreter sets the __name__ variable to have a value “__main__�?
# If this file is being imported from another module, __name__ will be set to the module’s name.
# if __name__ == '__main__':
# 	app.run(debug=True) # turn off if production mode

sample = {
	'INTAKE TERM CODE': 2020,
	'ADMIT TERM CODE': 2020,
	'INTAKE COLLEGE EXPERIENCE': 'New to College',
	'PRIMARY PROGRAM CODE': 9111,
	'SCHOOL CODE': 'CH',
	'STUDENT LEVEL NAME': 'Post Secondary',
	'MAILING POSTAL CODE GROUP 3': 'L1V',
	'GENDER': 'F',
	'DISABILITY IND': 'N',
	'FUTURE TERM ENROL': '1-1-1-1-0-0-0-0-0-0',
	'ACADEMIC PERFORMANCE': 'DF - Poor',
	'EXPECTED GRAD TERM CODE': 2020,
	'FIRST YEAR PERSISTENCE COUNT': 0,
	'HS AVERAGE MARKS': 0,
	'ENGLISH TEST SCORE': 140,
	'AGE GROUP LONG NAME': '41 to 50',
	'FIRST GENERATION IND': 'Y',
	'effective academic history': 'no',
	'APPLICANT TARGET SEGMENT NAME': 'Non-Direct Entry',
	'TIME STATUS NAME': 'Full-Time',
	'FUNDING SOURCE NAME': 'GPOG - FT'
	}

sample = pd.DataFrame([sample])
print("FORMATED FORM (ready to transform into a DF)")
print("--------------------------------------------")
print(type(sample))
print(sample)
print("\n")

# encoding categorical features
categorical_column = [col for col in sample.columns if sample[col].dtype == 'object'] + ['PRIMARY PROGRAM CODE', 'INTAKE TERM CODE', 'ADMIT TERM CODE', 'EXPECTED GRAD TERM CODE']
for column in categorical_column:
	encoder = label_encoders[column]
	print(column + ':')
	print(encoder.classes_)
	sample[column] = pd.Series(encoder.transform(sample[column]))
# feature reduction
reduced_feature = pca.transform(sample[['FUNDING SOURCE NAME', 'TIME STATUS NAME']])  # FUNDING SOURCE NAME is highly correlated with TIME STATUS NAME
sample.insert(6, 'time and fund', reduced_feature)
sample.drop(columns=['FUNDING SOURCE NAME', 'TIME STATUS NAME'], inplace=True)
prediction = model.predict(sample)[:, 1][0]

if prediction < .5:
	'%.2f % to fail. well done! keep hard working'
else:
	'%.2f % to fail.'