import re
from tabulate import tabulate
import operator
import math
import sys

OUTPUT_FILE = 'adult.out'

#attributes to be ignored in the classifier (excluding class because I explicitly gather it) 
EXCLUDED_ATTRIBUTES = ['class', 'education-num', 'fnlwgt']
#how to discretize the numeric features 
# >1 specifies how many buckets you want, <1 specifics info gain threshold, 'none' means leave numeric, [list] means expert makes buckets
DISCRETE_TYPE = {'age': 'none', 'hours-per-week': 'none', 'capital-gain' : 0.02, 'capital-loss': 'none'}

def control(file_path):
	"""
	Controls the process of reading in the data, 
	replacing missing values, and discretizing 
	numeric features. 
	
	Args:
		file_path (string) : the path to the dataset file

	Returns:
		data (dict) : the complete dataset
						data[record][feature][unique_value]
		features (dict) : holds features and  their unique values
				   	 features[feature] = [list unique values]
				   	 or
				   	 features[feature] = 'Numeric' 
		totals (dict) : dataset breakdown by class
						totals = {>50K : #, <=50K: #}
	"""
	global data 
	global categorical_dict

	#read in the data file
	averages, totals = read_data(file_path)
	#find the averages/modes for each feature
	mode = find_avg_mode(averages)
	#display the averages/modes to the user 
	print_feature_averages_by_class(mode, totals)
	#replace missing values in data set with averages/modes
	replace_missing_values(mode)

	#build list of numeric features which need to be discretized
	numeric_list = []
	for feature in attribute_dict:
		if feature not in categorical_dict:
			numeric_list.append(feature)

	#discretize all the numeric features
	counts = {}
	print "DISCRETIZATION"
	with open("adult.out", "a") as out:
		out.write('\nDISCRETIZATION\n')
	for feature in numeric_list:
		#run the discretization 
		discretized, lower, upper = run_entropy(feature, DISCRETE_TYPE[feature])
		#change the stored dataset to reflect the new categories
		counts = update_data(discretized, lower, upper, feature, counts)

	features = categorical_dict

	return data, features, totals


def read_data(file_path):
	"""
	Reads in the data from the file.
	Builds feature list from file info.
	Calls entry method based on numeric or 
	categorical.
	
	Args:
		file_path (string) : the path to the dataset file

	Returns:
		averages (dict) : the sum/count for numeric and count
				   for categorical collected during entry phase
		totals (dict) : the total number of records by class
					totals[class] = number of records
	"""

	global data
	data = {}
	global categorical_dict
	categorical_dict = {}
	global attribute_dict
	attribute_dict = {}

	averages = {}

	#prepare dictionary to collect the class counts
	totals = {'>50K': 0, '<=50K': 0}

	#open the data file and read it in line by line
	with open(file_path, 'r') as data_file:
		#label each record by an entry number
		entry_number = 0
		#count how many features in order to know idx
		count_features = 0

		for line in data_file:
			#check if the line is explaining and attribute/feature
			if re.search('@attribute', line):
				line = line.replace('\n', '')
				line = line.replace(',', '')
				#this split will put feature name at idx 1
				attribute_info = line.split(' ')

				if attribute_info[1] not in EXCLUDED_ATTRIBUTES:
					#add all the attributes to a dictionary of attribute:idx
					attribute_dict[attribute_info[1]] = count_features

					#for features with {} will we gather the valid categories
					#and add them to the specific dictionary or categorical features
					if '{' in attribute_info: 
						unique_list = []
						for x in range(3, len(attribute_info)-1):
							unique_list.append(attribute_info[x])
						categorical_dict[attribute_info[1]] = unique_list
				count_features += 1
			#check if the data is starting and prepare dict to hold averages
			elif '@data' in line:
				averages = helper_averages()
			#skip blank lines
			elif ',' not in line:
				pass
			#enters the data
			else:
				entry_number += 1
				line = line.replace(' ', '')
				line = line.replace('\n', '')
				attributes = line.split(',')

				#first store the class type for the record
				class_type = enter_class(entry_number, attributes[14])
				totals[class_type] += 1

				#iterate through the features and enter the data
				for feature in attribute_dict:
					if feature not in categorical_dict:
						averages = enter_numeric_feature(averages, entry_number, feature, attributes[attribute_dict[feature]], class_type)
					else:
						averages = enter_categorical_feature(averages, entry_number, feature, attributes[attribute_dict[feature]], categorical_dict[feature], class_type)
	return averages, totals


def enter_class(entry_number, str_class):
	"""
	Takes string of >50K (C1) or <=50K (C2) 
	and assigns to class and stores in data. 
	
	Args:
		entry_number (int): record number that identifies
					   the specific sample in data
		str_class (string) : from file, tells class

	Returns:
		class_label (string) : class id, either >50K or <=50K
	"""

	global data

	if '<=50K' in str_class:
		class_label = '<=50K'
	elif '>50K' in str_class:
		class_label = '>50K'
	#because NB is supervised, need to know if record does not have label
	else:
		print str(entry_number) + ' does not have an asssigned class label.'

	data[str(entry_number)] = {'class': class_label}

	return class_label


def enter_numeric_feature(averages, entry_number, feature, str_unique, class_type):
	"""
	Enters numeric features in data. 
	Updates the sum and count in averages.
	Invalid ages and blank values replaced with '?'
	
	Args:
		averages (dict) : holds sum/count for features
		entry_number (int) : record number that identifies
					   the specific sample in data
		feature (string) : feature being entered
		str_unique (string) : unique feature value for record 
		class_type (string) : either >50K or <=50K

	Returns:
		averages (dict) : updated sum/count for features
	"""

	global data

	if '?' in str_unique:
		data[str(entry_number)][feature] = '?'
	elif feature == 'age' and (int(str_unique) <= 0 or int(str_unique) > 120):
		data[str(entry_number)]['age'] = '?'
	else:
		data[str(entry_number)][feature] = int(str_unique)
		averages[feature][class_type]['sum'] += int(str_unique)
		averages[feature][class_type]['count'] += 1
		
	return averages


def enter_categorical_feature(averages, entry_number, feature, str_unique, valid_list, class_type):
	"""
	Enters categorical features in data. 
	Updates the count for unique feature averages.
	Invalid and blank values replaced with 'Invalid'
	
	Args:
		averages (dict) : holds sum/count for features
		entry_number (int) : record number that identifies
					   		the specific sample in data
		feature (string) : feature being entered
		str_unique (string) : unique feature value for record 
		valid_list (list strings) : list of the valid values for that categorical feature
		class_type (string) : either >50K or <=50K

	Returns:
		averages (dict) : updated sum/count for features
	"""

	global data

	if str_unique in valid_list:
		data[str(entry_number)][feature] = str_unique
		averages[feature][class_type][str_unique] += 1
	else:
		data[str(entry_number)][feature] = 'Invalid' 

	return averages


def find_avg_mode(averages):
	"""
	Finds mean for numeric features 
	and modes for categorical features
	by class.
	
	Args:
		averages (dict) : holds sum/count for features

	Returns:
		mode (dict) : holds mean/mode for each feature
	"""
	
	#categorical holds all attributes that are categorical
	global categorical_dict
	#attribute holds label of all features (numeric and categorical)
	global attribute_dict

	#construct a dictionary to hold the mode/mean values
	mode = {}
	for feature in attribute_dict:
		mode[feature] = {'>50K': 0, '<=50K': 0}

	for feature in attribute_dict:
		#calculates the mean for numeric features by class
		if feature not in categorical_dict:
			mode[feature]['>50K'] = str(int(round(float(averages[feature]['>50K']['sum'])/averages[feature]['>50K']['count'])))
			mode[feature]['<=50K'] = str(int(round(float(averages[feature]['<=50K']['sum'])/averages[feature]['<=50K']['count'])))
		#calculates the mode for categorical features by class
		else:
			for class_type in averages[feature]:
				maximum = 0
				current = 0
				for unique in averages[feature][class_type]:
					current = averages[feature][class_type][unique]
					if current > maximum:
						mode[feature][class_type] = unique
						maximum = current

	return mode


def print_feature_averages_by_class(mode, totals):
	"""
	Prints out information on how many records by class.
	Prints out averages/modes of each feature by class. 
	
	Args:
		mode (dict) : holds mean/mode for each feature
		totals (dict) : total number of records by class
				 		totals[class] = number of records
	Returns:
		none
	"""

	global attribute_dict

	#print the class breakdown of the dataset
	header = ['>50k', '<=50k', 'Total']
	row = [totals['>50K'], totals['<=50K'], (totals['>50K'] + totals['<=50K'])]
	print "\nEntire Dataset"
	print tabulate([header, row], headers="firstrow", tablefmt = 'grid')

	with open("adult.out", "a") as out:
		out.write('Entire Dataset\n')
		out.write(tabulate([header, row], headers="firstrow", tablefmt = 'grid'))

	c1 = []
	c2 = []
	list_features = []
	
	header = sorted(attribute_dict, key=attribute_dict.__getitem__)
	c1.append('>50K')
	c2.append('<=50K')
	#build ordered list for each class for printing ease 
	for feature in header:
		c1.append(mode[feature]['>50K'])
		c2.append(mode[feature]['<=50K'])
	#print the table with the feature averages/modes by class
	print '\nMEAN/MODE of each feature by class.'
	print tabulate([header, c1, c2], headers='firstrow', tablefmt='grid')

	with open("adult.out", "a") as out:
		out.write('\n\nMEAN/MODE of each feature by class.\n')
		out.write(tabulate([header, c1, c2], headers='firstrow', tablefmt='grid'))
		out.write('\n')


def replace_missing_values(mode):
	"""
	Replaces any 'Invalid' or '?' with the mean 
	or mode for that feature with that class label.
	
	Args:
		mode (dict) : holds mean/mode for each feature

	Returns:
		none
	"""

	global data

	#variables to keep track of how many values need to be replaced
	cnt_sample = 0
	cnt_replace = 0
	missing = False

	#iterate through samples and replace with mean/mode if invalid or ?
	for sample in data:
		for feature in data[sample]:
			if data[sample][feature] == '?' or data[sample][feature] == 'Invalid':
				data[sample][feature] = mode[feature][data[sample]['class']]
				cnt_replace += 1
				missing = True
		if missing:
			cnt_sample += 1
			missing = False

	#display how many records had to be updated with mean/mode
	print (str(cnt_sample) + ' records had missing feature values. ' + str(cnt_replace) + ' feature values were replaced with mean/mode value for the class.\n')
	with open("adult.out", "a") as out:
		out.write((str(cnt_sample) + ' records had missing feature values.' + '\n'))
		out.write(str(cnt_replace) + ' feature values were replaced with mean/mode value for the class.\n')


def run_entropy(feature, type_term):
	"""
	Discretizes numeric features. 
	It is the wrapper for the recursive function.
	
	Args:
		feature (string) : feature that will be discretized
		type_term (list, int, string) : type of termination condition
					< 1 = terminate when falls below 
							specific information gain, 
							value is info gain condition
					> 1 = terminate by buckets, value is 
							how many buckets
					'none' = leave numeric
					[int] = expert defined bucket

	Returns:
		split_value_array (list) : values to build the 
									numeric categories on
		lower (int) : lowest value in the feature 
		upper (int) : highest value in the feature
	"""

	global data
	global split_value_array


	#list that holds sorted unique values for the numeric feature
	attribute_values = []

	#Step 1: Sort the data from low to high 
	for sample in data:
		#only store a number once in the array to save efficiency
		if data[sample][feature] not in attribute_values:
			attribute_values.append(data[sample][feature])
	attribute_values = sorted(attribute_values)

	#if the buckets have already been defined by expert
	if type_term == 'none':
		return 'none', 0, attribute_values[-1]
	elif type(type_term) == list:
		return discretize_byexpert(feature, attribute_values, type_term)
		


	#build a matrix to store values every time a partition is assessed 
	matrix = {'<=split': {'>50K': 0, '<=50K': 0, 'total': 0}, '>split': {'>50K': 0, '<=50K': 0, 'total': 0}}


	#Step 2: Find the initial entropy of the whole dataset
	num_c1 = float(0)
	num_c2 = float(0)
	for sample in data:
		if data[sample]['class'] == '>50K':
			num_c1 += 1
		elif data[sample]['class'] == '<=50K':
			num_c2 += 1
	entropy_C1 = (num_c1/(num_c1+num_c2)) * math.log((num_c1/(num_c1+num_c2)), 2)
	entropy_C2 = (num_c2/(num_c1+num_c2)) * math.log((num_c2/(num_c1+num_c2)), 2)
	initial_entropy = -(entropy_C1 + entropy_C2)

	split_value_array = []
	if type_term < 1:
		discretize_byinfogain(feature, attribute_values, initial_entropy, 0, attribute_values[-1], len(attribute_values), type_term)
	elif type_term > 1:
		discretize_bybucket(feature, attribute_values, initial_entropy, 0, attribute_values[-1], len(attribute_values), type_term)
		
	return split_value_array, 0, attribute_values[-1]


def discretize_bybucket(feature, attribute_values, initial_entropy, lower, upper, number, buckets):
	"""
	Discretizes numeric features into set number of buckets. 
	It is a recursive function.
	
	Args:
		feature (string) : feature that will be discretized
		attribute_values (list) :  unique numeric values for feature
		initial_entropy (float) : value of the entropy of previous breakdown
		lower (float) : lowest value of set to find partition
		upper (float) : highest value of set to find the partition
		number (int) : number of values in set (-1 will be num partitions)
		buckets (int) : number of buckets remaining to be made

	Returns:
		none - when returns begins to fold up until it reaches the wrapper
	"""

	global split_value_array

	# return if no more buckets to make or less than two values in set
	if (buckets == 1 or buckets == 0 or number <= 2):
		return

	#need a counter in case all the partitions achieve zero entropy, then return
	count_zero_entropy = 0
	num_partitions = number - 1
	matrix = {'<=split': {'>50K': 0, '<=50K': 0, 'total': 0}, '>split': {'>50K': 0, '<=50K': 0, 'total': 0}}

	#for loop builds matrix for class dist. above and below split value, finds info gain
	for x in range(0, num_partitions):
		#find the split_value by finding mean between the two numbers surronding partition
		split_value = float((attribute_values[x] + attribute_values[x+1])) / 2
		#run through the entire dataset 
		for sample in data:
			#check to make sure the value is within the bounds (important for recursion)
			if (int(data[sample][feature]) <= upper and int(data[sample][feature]) >= lower):
				if data[sample][feature] <= split_value and data[sample]['class'] == '>50K':
					matrix['<=split']['>50K'] += 1
					matrix['<=split']['total'] += 1
				elif data[sample][feature] <= split_value and data[sample]['class'] == '<=50K':
					matrix['<=split']['<=50K'] += 1
					matrix['<=split']['total'] += 1
				elif data[sample][feature] > split_value and data[sample]['class'] == '>50K':
					matrix['>split']['>50K'] += 1
					matrix['>split']['total'] += 1
				elif data[sample][feature] > split_value and data[sample]['class'] == '<=50K':
					matrix['>split']['<=50K'] += 1
					matrix['>split']['total'] += 1
		#A*logA + B*logB = Entropy_below_split
		#C*logC + D*logD = Entropy_above_split
		A = (float(matrix['<=split']['>50K'])/matrix['<=split']['total'])
		B = (float(matrix['<=split']['<=50K'])/matrix['<=split']['total'])
		C = (float(matrix['>split']['>50K'])/matrix['>split']['total'])
		D = (float(matrix['>split']['<=50K'])/matrix['>split']['total'])
		
		#cannot take a log base 2 of 0, the answer is -infinity
		if A == 0:
			entropy_below = -(B * math.log(B, 2))
		elif B == 0:
			entropy_below = -(A * math.log(A, 2))
		else:
			entropy_below = -(A * math.log(A, 2) + B * math.log(B, 2))
		if C == 0:
			entropy_above = -(D * math.log(D, 2))
		elif D == 0:
			entropy_above = -(C * math.log(C, 2))
		else:
			entropy_above = -(C * math.log(C, 2) + D * math.log(D, 2))

		#net_entropy = (num below split/total number) * Entropy below + (num above split/total number) * Entropy above
		net_entropy = (float(matrix['<=split']['total'])/(matrix['<=split']['total'] + matrix['>split']['total'])) * entropy_below
		net_entropy = net_entropy + (float(matrix['>split']['total'])/(matrix['<=split']['total'] + matrix['>split']['total'])) * entropy_above

		#calculate the improvement in the entropy, or the information gained
		info_gain = (initial_entropy - net_entropy)
		if net_entropy == 0:
			count_zero_entropy += 1
		#if this is the first partition, make this the standard to beat
		if x == 0:
			max_gain = info_gain
			max_split = split_value
		#update the max information if a better information gain reached
		if info_gain > max_gain:
			max_gain = info_gain
			max_split = split_value

	#if all the partitions gave 0 entropy, return and don't partition further
	if count_zero_entropy == num_partitions:
		return

	#append the split value to the array to store
	split_value_array.append(max_split)
	
	#build list of those numbers below and above the split to send to the recursion
	lower_att = []
	higher_att = []
	for att in attribute_values:
		if att <= max_split:
			lower_att.append(att)
		else:
			higher_att.append(att)

	#decide how to divide the buckets, if odd, add more to bigger side
	bucket_lower = buckets/2
	bucket_upper = buckets/2
	if buckets % 2 == 1:
		if len(higher_att) > len(lower_att):
			bucket_upper += 1
		else:
			bucket_lower += 1

	#recursive through the lower part and the upper part
	discretize_bybucket(feature, lower_att, (initial_entropy - max_gain), lower, max_split, len(lower_att), bucket_lower)
	discretize_bybucket(feature, higher_att, (initial_entropy - max_gain), max_split, upper, len(higher_att), bucket_upper)
	return


def discretize_byinfogain(feature, attribute_values, initial_entropy, lower, upper, number, threshold):
	"""
	Discretizes numeric features and terminates by info gain threshold. 
	It is a recursive function.
	
	Args:
		feature (string) : feature that will be discretized
		attribute_values (list) :  unique numeric values for feature
		initial_entropy (float) : value of the entropy of previous breakdown
		lower (float) : lowest value of set to find partition
		upper (float) : highest value of set to find the partition
		number (int) : number of values in set (-1 will be num partitions)
		thresholds (float) : info gain needed in order to further recurse

	Returns:
		none - when returns begins to fold up until it reaches the wrapper
	"""

	global split_value_array

	# return if less than two values in set
	if number <= 2:
		return

	#need a counter in case all the partitions achieve zero entropy, then return
	count_zero_entropy = 0
	num_partitions = number - 1
	matrix = {'<=split': {'>50K': 0, '<=50K': 0, 'total': 0}, '>split': {'>50K': 0, '<=50K': 0, 'total': 0}}

	#for loop builds matrix for class dist. above and below split value, finds info gain
	for x in range(0, num_partitions):
		#find the split_value by finding mean between the two numbers surronding partition
		split_value = float((attribute_values[x] + attribute_values[x+1])) / 2
		#run through the entire dataset 
		for sample in data:
			#check to make sure the value is within the bounds (important for recursion)
			if (int(data[sample][feature]) <= upper and int(data[sample][feature]) >= lower):
				if data[sample][feature] <= split_value and data[sample]['class'] == '>50K':
					matrix['<=split']['>50K'] += 1
					matrix['<=split']['total'] += 1
				elif data[sample][feature] <= split_value and data[sample]['class'] == '<=50K':
					matrix['<=split']['<=50K'] += 1
					matrix['<=split']['total'] += 1
				elif data[sample][feature] > split_value and data[sample]['class'] == '>50K':
					matrix['>split']['>50K'] += 1
					matrix['>split']['total'] += 1
				elif data[sample][feature] > split_value and data[sample]['class'] == '<=50K':
					matrix['>split']['<=50K'] += 1
					matrix['>split']['total'] += 1
		#A*logA + B*logB = Entropy_below_split
		#C*logC + D*logD = Entropy_above_split
		A = (float(matrix['<=split']['>50K'])/matrix['<=split']['total'])
		B = (float(matrix['<=split']['<=50K'])/matrix['<=split']['total'])
		C = (float(matrix['>split']['>50K'])/matrix['>split']['total'])
		D = (float(matrix['>split']['<=50K'])/matrix['>split']['total'])
		
		#cannot take a log base 2 of 0, the answer is -infinity
		if A == 0:
			entropy_below = -(B * math.log(B, 2))
		elif B == 0:
			entropy_below = -(A * math.log(A, 2))
		else:
			entropy_below = -(A * math.log(A, 2) + B * math.log(B, 2))
		if C == 0:
			entropy_above = -(D * math.log(D, 2))
		elif D == 0:
			entropy_above = -(C * math.log(C, 2))
		else:
			entropy_above = -(C * math.log(C, 2) + D * math.log(D, 2))

		#net_entropy = (num below split/total number) * Entropy below + (num above split/total number) * Entropy above
		net_entropy = (float(matrix['<=split']['total'])/(matrix['<=split']['total'] + matrix['>split']['total'])) * entropy_below
		net_entropy = net_entropy + (float(matrix['>split']['total'])/(matrix['<=split']['total'] + matrix['>split']['total'])) * entropy_above
		
		#calculate the improvement in the entropy, or the information gained
		info_gain = (initial_entropy - net_entropy)
		if net_entropy == 0:
			count_zero_entropy += 1
		#if this is the first partition, make this the standard to beat
		if x == 0:
			max_gain = info_gain
			max_split = split_value
		#update the max information if a better information gain reached
		if info_gain > max_gain:
			max_gain = info_gain
			max_split = split_value

	#if all the partitions gave 0 entropy, return and don't partition further
	if count_zero_entropy == num_partitions:
		return
	#recurse only if gain is over threshold
	if max_gain >= threshold:
		#append the split value to the array to store
		split_value_array.append(max_split)
		#build list of those numbers below and above the split to send to the recursion
		lower_att = []
		higher_att = []
		for att in attribute_values:
			if att <= max_split:
				lower_att.append(att)
			else:
				higher_att.append(att)

		#recursive through the lower part and the upper part
		discretize_byinfogain(feature, lower_att, (initial_entropy - max_gain), lower, max_split, len(lower_att), threshold)
		discretize_byinfogain(feature, higher_att, (initial_entropy - max_gain), max_split, upper, len(higher_att), threshold)
	return


def discretize_byexpert(feature, attribute_values, buckets):
	"""
	Discretizes numeric features and terminates by info gain threshold. 
	It is a recursive function.
	
	Args:
		feature (string) : feature that will be discretized
		attribute_values (list) :  unique numeric values for feature
		buckets (list int) : values to divide on

	Returns:
		split_value_array (list) : values to build the 
									numeric categories on
		lower (int) : lowest value in the feature 
		upper (int) : highest value in the feature
	"""


	return buckets, attribute_values[0], attribute_values[-1]


def update_data(discretized, lower, upper, feature, counts):
	"""
	Builds categories for numerical feature based on 
	list of discretized. Replaces value in data table
	with correct category.
	
	Args:
		discretized (list) : split values to build categories from
		lower (int) : lowest value of set to find partition
		upper (int) : highest value of set to find the partition
		feature (string) : feature that will be discretized
		counts (dict) : how many samples in each category

	Returns:
		counts (dict) : updated of how many samples in each category
	"""

	global data
	global categorical_dict

	categories = {}
	counts[feature] = {}
	
	if discretized == 'none':
		categorical_dict[feature] = 'Numeric'
		return counts 

	if not discretized:
		categorical_dict[feature] = 'Numeric'
		print (feature + ' failed to pass the information gain threshold and was not discretized.')
		with open("adult.out", "a") as out:
			out.write(feature + ' failed to pass the information gain threshold and was not discretized.')
			out.write('\n')
		return counts 


	#sort the discretized so they run low to high
	if type(discretized) != float:
		discretized = sorted(discretized)
	else:
		discretized = [discretized]

	#build the categories in string form
	for x in range(0,len(discretized) + 1):
		if x == 0:
			categories[discretized[x]] = (str(lower) + '-' + str(discretized[x]))
		elif x == (len(discretized)):
			categories[upper] = (str(discretized[x-1]) + '-' + str(upper))
		else:
			categories[discretized[x]] = (str(discretized[x-1]) + '-' + str(discretized[x]))

	#add the feature and categories to the categorical dict
	categorical_dict[feature] = []
	for bucket in categories:
		categorical_dict[feature].append(categories[bucket])
		counts[feature][categories[bucket]] = 0

	#update the value in each record to be the right category instead of numeric]
	for record in data:
		assigned = False
		for idx, split_value in enumerate(discretized):
			if not assigned:
				if idx == 0 and split_value == discretized[-1]:
					if data[record][feature] <= discretized[idx]:
						data[record][feature] = categories[split_value]
						counts[feature][categories[split_value]] += 1
						assigned = True
					else:
						data[record][feature] = categories[upper]
						counts[feature][categories[upper]] += 1
						assigned = True
				elif idx == 0:
					if data[record][feature] <= discretized[idx]:
						data[record][feature] = categories[split_value]
						counts[feature][categories[split_value]] += 1
						assigned = True
				elif (data[record][feature] <= discretized[idx] and data[record][feature] > discretized[idx-1]):
					data[record][feature] = categories[split_value]
					counts[feature][categories[split_value]] += 1
					assigned = True
				elif split_value == discretized[-1]:
					data[record][feature] = categories[upper]
					counts[feature][categories[upper]] += 1
					assigned = True

	display_discretize(counts, categories, feature)
	
	return counts


def display_discretize(counts, categories, feature):
	"""
	Prints out information on the categories created by
	discretizing. Please note NOT DYNAMIC. NEED TO 
	MANUALLY CHANGE IF BUCKET NUMBER CHANGES.
	
	Args:
		counts (dict) : how many samples in each category
		categories (dict): the <split value - category> for sorting 
	Returns:
		none
	"""

	header = []
	row = []

	keys_sorted = sorted(categories.keys())
	header.append(feature)
	row.append('# samples')
	for key in keys_sorted:
		header.append(categories[key]) 
		row.append(counts[feature][categories[key]])
	print tabulate([header, row], headers="firstrow", tablefmt = "grid")
	with open("adult.out", "a") as out:
		out.write(tabulate([header, row], headers="firstrow", tablefmt = "grid"))
		out.write('\n')


def helper_averages():
	"""
	Builds dictionary to hold statistics as data is entered
	to be used for averages/modes.
	
	Args:
		none

	Returns:
		averages (dict) : the sum/count for numeric and count
				   for categorical collected during entry phase
	"""

	global categorical_dict
	global attribute_dict

	averages = {}

	for feature in attribute_dict:
		if feature not in categorical_dict:
			averages[feature] = {'>50K': {'sum':0, 'count':0}, '<=50K': {'sum':0, 'count':0}}
		else:
			averages[feature] = {'>50K': {}, '<=50K': {}}
			for unique in categorical_dict[feature]:
				averages[feature]['>50K'][unique] = 0
				averages[feature]['<=50K'][unique] = 0
	return averages



