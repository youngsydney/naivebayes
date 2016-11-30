import math
import numpy
from tabulate import tabulate
import sys

OUTPUT_FILE = 'adult.out'


def run_10_fold(original_data, totals, features):
	"""
	Controls the running of 10-fold cross validaton 
	on Naive Bayes modeler and the dataset.

	Args:
		original_data (dict) : complete dataset
		total (dict) :the class counts
		features (dict) : the features and
					their valid options (if categorical)
					or the feature and 'Numeric' 

	Returns:
		none
	"""
	global metrics
	metrics = {}

	#breakup the dataset into ten subsets
	folds = select_10_folds_stratified(original_data)

	#find and display the baseline data
	classified_samples = baseline_data(original_data, totals)
	confusion_matrix = evalute(classified_samples, original_data, 'Baseline')
	evaluate_results(confusion_matrix, 'Baseline')

	#build and model, test, and evalute on each fold
	for x in range(1,11):
		#create a training and a test dataset
		training_dataset = original_data.copy()
		test_dataset = original_data.copy()
		#remove the selected fold from the training
		#remove all non selected from test
		for record_number in original_data:
			if record_number in folds[str(x)]:
				del training_dataset[record_number]
			else:
				del test_dataset[record_number]

		#build the model and return all the precalculated probabilities
		prob_matrix, tr_totals = build_model(training_dataset, features)
		classified_samples = classify_data(prob_matrix, test_dataset, tr_totals)
		confusion_matrix = evalute(classified_samples, test_dataset, str(x))
		evaluate_results(confusion_matrix, str(x))
	evaluation_averages()
	print_evaluation_results()


def select_10_folds_stratified(data):
	"""
	Portions the dataset into 10 separate blocks
	that are stratified in class type.

	Args:
		data (dict) : complete dataset

	Returns:
		folds (dict) : the different folds
				<fold_#: data, fold_#: data...>
	"""

	data_c1 = []
	data_c2 = []

	#split the data into two lists of record numbers
	for sample in data:
		if data[sample]['class'] == '>50K':
			data_c1.append(sample)
		else:
			data_c2.append(sample)

	folds = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []}
	#iterate through and use a counter to assign to fold
	count = 1
	for record in data_c1:
		folds[str(count)].append(record)
		if count == 10:
			count = 1
		else:
			count += 1
	count = 1
	for record in data_c2:
		folds[str(count)].append(record)
		if count == 10:
			count = 1
		else:
			count += 1

	return folds


def select_10_folds(data):
	"""
	Portions the dataset into 10 separate blocks.
	Done at random.

	Args:
		data (dict) : complete dataset

	Returns:
		folds (dict) : the different folds
				<fold_#: data, fold_#: data...>
	"""

	folds = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []}

	#iterate through and use a counter to assign to fold
	count = 1
	for record in data:
		folds[str(count)].append(record)
		if count == 10:
			count = 1
		else:
			count += 1
	count = 1
	return folds


def build_model(data, features):
	"""
	Builds a Naive Bayes model using the 
	training dataset. Stores all the 
	probabilities.

	Args:
		data (dict) : training dataset
		features (dict) : the features and
					their valid options

	Returns:
		none
	"""
	
	prob_matrix = helper_build_matrix(features)

	totals = {'>50K': 0, '<=50K': 0, 'total': 0}

	#iterate through every sample in the dataset
	for sample in data:
		totals[data[sample]['class']] += 1
		totals['total'] += 1
		#iterate through all the features
		for feature in features:
			#handle numeric -- build a list of all the values
			if features[feature] == 'Numeric':
				if data[sample]['class'] in prob_matrix[feature]:
					prob_matrix[feature][data[sample]['class']]['list'].append(data[sample][feature])
				else:
					prob_matrix[feature][data[sample]['class']] = {'list': [], 'mean': 0, 'stdv': 0}
					prob_matrix[feature][data[sample]['class']]['list'].append(data[sample][feature])
			#handle categorical -- count sample with each unique feature value
			else:
				#count how many of that unique value in that class
				prob_matrix[feature][data[sample][feature]][data[sample]['class']] += 1
				#count how many total records have that unique value
				prob_matrix[feature][data[sample][feature]]['total'] += 1


	#calculate all the probabilties for the unique feature values by class
	for feature in prob_matrix:
		if features[feature] == 'Numeric':
			prob_matrix[feature]['>50K']['mean'] = float(sum(prob_matrix[feature]['>50K']['list']))/len(prob_matrix[feature]['>50K']['list'])
			prob_matrix[feature]['<=50K']['mean'] = float(sum(prob_matrix[feature]['<=50K']['list']))/len(prob_matrix[feature]['<=50K']['list'])
			prob_matrix[feature]['>50K']['stdv'] = st_dev(prob_matrix[feature]['>50K']['list'])
			prob_matrix[feature]['<=50K']['stdv'] = st_dev(prob_matrix[feature]['<=50K']['list'])
		else:
			for unique in prob_matrix[feature]:
				prob_matrix[feature][unique]['>50K'] = (float(prob_matrix[feature][unique]['>50K'] + 0.5)/(totals['>50K'] + 1))
				prob_matrix[feature][unique]['<=50K'] = (float(prob_matrix[feature][unique]['<=50K'] + 0.5)/(totals['<=50K'] + 1))
	return prob_matrix, totals


def classify_data(prob_matrix, test_data, totals):
	"""
	Controls the running of 10-fold cross validaton 
	on Naive Bayes modeler and the dataset.

	Args:
		original_data (dict) : complete dataset, all
						categorized at this point
		total (dict) : the class counts
		features (dict) : the features and
					their valid options

	Returns:
		classified_samples (dict) : entry numbers
								and their predicted class
	"""
	post_prob_C1 = 0
	post_prob_C2 = 0

	classified_samples = {}

	#iterate through everyone of the test samples
	for sample in test_data:
		#for every feature in the sample calculate prob C1 and C2
		for feature in test_data[sample]:
			if feature != 'class':
				if '<=50K' in prob_matrix[feature]:
					#(1/sqrt(2*pi*stdv)) * e ^ (-(x-mean)^2/(2*stdv^2))
					# A = sqrt(2*pi*stdv)
					# B = -(x-mean)^2/(2*stdv^2)
					A = (1/math.sqrt(2*math.pi*prob_matrix[feature]['>50K']['stdv']))
					B = (-math.pow((test_data[sample][feature] - prob_matrix[feature]['>50K']['mean']), 2)) / (2 * math.pow(prob_matrix[feature]['>50K']['stdv'], 2))
					post_prob_C1 += A * math.pow(2.718281, B)
					A = (1/math.sqrt(2*math.pi*prob_matrix[feature]['<=50K']['stdv']))
					B = (-math.pow((test_data[sample][feature] - prob_matrix[feature]['<=50K']['mean']), 2)) / (2 * math.pow(prob_matrix[feature]['<=50K']['stdv'], 2))
					post_prob_C2 += A * math.pow(2.718281, B)
				else:
					post_prob_C1 *= prob_matrix[feature][test_data[sample][feature]]['>50K']
					post_prob_C2 *= prob_matrix[feature][test_data[sample][feature]]['<=50K']
		post_prob_C1 = post_prob_C1 * (float(totals['>50K'])/(totals['>50K'] + totals['<=50K']))
		post_prob_C2 = post_prob_C2 * (float(totals['<=50K'])/(totals['>50K'] + totals['<=50K']))
		if post_prob_C1 > post_prob_C2:
			classified_samples[sample] = '>50K'
		else:
			classified_samples[sample] = '<=50K'
	return classified_samples


def evalute(classified_samples, test_dataset, fold):
	"""
	Builds the confusion matrix for the given model.
	>50K is Positive and <=50K is negative

	Args:
		classified_samples (dict) : records and the class
									they were assigned
		test_dataset (dict) : the set of tested data with 
							correct labels
		fold (int) : the number fold/model being testing

	Returns:
		confusion_matrix (dict) : the matrix of TP/FP/TN/FN 
								for the given model
	"""
	
	confusion_matrix = {'TP':0, 'FP':0, 'TN':0, 'FN':0}

	for record in classified_samples:
		if classified_samples[record] == test_dataset[record]['class']:
			if classified_samples[record] == '>50K':
				confusion_matrix['TP'] += 1
			else:
				confusion_matrix['TN'] += 1
		if classified_samples[record] != test_dataset[record]['class']:
			if classified_samples[record] == '>50K':
				confusion_matrix['FP'] += 1
			else:
				confusion_matrix['FN'] += 1

	display_confusion_matrix(confusion_matrix, fold)
	return confusion_matrix
	

def display_confusion_matrix(confusion_matrix, fold):
	"""
	Builds the confusion matrix for the given model.
	>50K is Positive and <=50K is negative

	Args:
		confusion_matrix (dict) : the matrix of TP/FP/TN/FN 
								for the given model
		fold (int) : the number fold/model being testing

	Returns:
		none
	"""

	global metrics

	row1 = ['-', 'Predicted', '-']
	row2 = ['Actual', '>50K', '<=50K']
	row3 = ['>50K', confusion_matrix['TP'], confusion_matrix['FN']]
	row4 = ['<=50K', confusion_matrix['FP'], confusion_matrix['TN']]

	print ('\nConfusion Matrix for Model ' + str(fold))
	print tabulate([row1, row2, row3, row4],tablefmt = "grid")
	with open("adult.out", "a") as out:
		out.write(('\nConfusion Matrix for Model ' + str(fold) + '\n'))
		out.write(tabulate([row1, row2, row3, row4],tablefmt = "grid"))
		out.write('\n\n')


def baseline_data(original_data, totals):
	"""
	Controls the baseline classification.
	If all samples assigned to class with 
	highest overall probability. 

	Args:
		original_data (dict) : complete dataset, all
						categorized at this point
		total (dict) : the class counts

	Returns:
		classified_samples (dict) : entry numbers
								and their predicted class
	"""
	global metrics

	#find the class with the greatest overall probability
	max_samples = 0
	winning_class = ''
	for class_type in totals:
		current_samples = totals[class_type]
		if current_samples > max_samples:
			winning_class = class_type
			max_samples = current_samples

	classified_samples = {}

	#iterate through everyone of the test samples
	for sample in original_data:
		classified_samples[sample] = winning_class
	return classified_samples


def helper_build_matrix(features):
	"""
	Builds a matrix to hold all the probability
	values for the model. 

	Args:
		features (dict) : the features and
					their valid options

	Returns:
		matrix (dict) : holds all probabilities
	"""

	matrix = {}
	for feature in features:
		matrix[feature] = {}
		for unique in features[feature]:
			matrix[feature][unique] = {'>50K': 0, '<=50K': 0, 'total': 0 }
	return matrix


def evaluate_results(cm, fold):
	"""
	Calculates the evaluation metrics for the fold

	Args:
		cm (dict) : (confusion matrix) the TP, TN, FP, FN
		fold (int) : the numer of the fold used for testing
	Returns:
		none
	"""
	global metrics
	metrics[fold] = {}


	totals = cm['TP']+ cm['TN'] + cm['FP']+ cm['FN']
	accuracy = str((float(cm['TP'])+ cm['TN'])/ totals)
	metrics[fold]['Accuracy'] = accuracy

	if (cm['TP']+ cm['FP']) != 0:
		ma_P_C1 = (float(cm['TP'])/ (cm['TP']+ cm['FP']))
	else: 
		ma_P_C1 = 0
	if (cm['TN']+ cm['FN']) != 0:
		ma_P_C2 = (float(cm['TN'])/ (cm['TN']+ cm['FN']))
	else:
		ma_P_C2 = 0
	ma_P = str((ma_P_C1 + ma_P_C2) / 2)
	metrics[fold]['Macro Precision'] = ma_P

	if (cm['TP']+ cm['FN']) != 0:
		ma_R_C1 = (float(cm['TP'])/ (cm['TP']+ cm['FN']))
	else:
		ma_R_C1 = 0
	if (cm['TN']+ cm['FP']):
		ma_R_C2 = (float(cm['TN'])/ (cm['TN']+ cm['FP']))
	else:
		ma_R_C2 = 0
	ma_R = str((ma_R_C1 + ma_R_C2) / 2)
	metrics[fold]['Macro Recall'] = ma_R

	if (ma_R_C1 + ma_P_C1) != 0:
		ma_F1_C1 = (2 * ma_R_C1 * ma_P_C1) / (ma_R_C1 + ma_P_C1)
	else:
		ma_F1_C1 = 0
	if (ma_R_C2 + ma_P_C2) != 0:
		ma_F1_C2 = (2 * ma_R_C2 * ma_P_C2) / (ma_R_C2 + ma_P_C2)
	else:
		ma_F1_C2 = 0
	ma_F1 = str((ma_F1_C1 + ma_F1_C2)/ 2)
	metrics[fold]['Macro F1'] = ma_F1

	mi_P = (float(cm['TP'] + cm['TN'])/ (cm['TP']+ cm['FP']+ cm['TN']+ cm['FN']))
	metrics[fold]['Micro Precision'] = mi_P
	mi_R = (float(cm['TP'] + cm['TN'])/ (cm['TP']+ cm['FN']+ cm['TN']+ cm['FP']))
	metrics[fold]['Micro Recall'] = mi_R
	mi_F1 = (2 * mi_R * mi_P) / (mi_R + mi_P)
	metrics[fold]['Micro F1'] = mi_F1

	if fold != 'Baseline':
		metrics[fold]['Models'] = ('Model ' + str(fold))
	else:
		metrics[fold]['Models'] = fold


def evaluation_averages():
	"""
	Finds the average accuracy over all the models.

	Args:
		none
	Returns:
		none
	"""

	global metrics
	metrics['Average'] = {}

	_sum_ = 0
	_count_ = 0 

	metrics['Average']['Models'] = 'Average'

	other = ['Micro Precision', 'Micro Recall', 'Micro F1', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Accuracy']
	for evaluation in other:
		for fold in metrics:
			if fold != 'Baseline' and fold != 'Average':
				_sum_ += float(metrics[fold][evaluation])
				_count_ += 1
		metrics['Average'][evaluation] = str(float(_sum_)/_count_)
		_sum_ = 0
		_count_ = 0


def print_evaluation_results():
	"""
	Prints the evaluation results for a single fold.

	Args:
		none
	Returns:
		none
	"""
	global metrics

	headers = ['Models', 'Micro Precision', 'Micro Recall', 'Micro F1', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Accuracy']
	f = {'Baseline': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], 'Average': []}
	for fold in f:
		for header in headers:
			f[fold].append(metrics[fold][header])

	print ('\nEVALUATION')
	print tabulate([headers, f['Baseline'], f['1'], f['2'], f['3'], f['4'], f['5'], f['6'], f['7'], f['8'], f['9'], f['10'], f['Average']], headers="firstrow", tablefmt = "grid")
	with open("adult.out", "a") as out:
		out.write(('\n\nEVALUATION\n'))
		out.write(tabulate([headers, f['Baseline'], f['1'], f['2'], f['3'], f['4'], f['5'], f['6'], f['7'], f['8'], f['9'], f['10'], f['Average']], headers="firstrow", tablefmt = "grid"))
		out.write('\n\n')


def st_dev(list_num):
	"""
	Calculates the standard deviation of a list.
	Args:
		list_num (list int) : set of values
	Returns:
		stdev (float) : standard deviation of set

	Algorithm adapted from: calebmadrigal.com/standard-deviation-in-python/
	"""

	avg = float(sum(list_num))/len(list_num)

	variance = map(lambda x: (x - avg)**2, list_num)

	st_dev = math.sqrt(float(sum(variance))/len(variance))
	return st_dev

















