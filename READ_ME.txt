READ_ME

Sydney Young (sey24)
February 18, 2016
COSC 285: Introduction to Data Mining
Project 1: Pre-Processing & Naive Bayes

How to Run: 
	On the command line -- navigate to the project directory and type "python run.py" to compile and run the run.py file.

	The program will run through to completion printing out relevant information along the way (recommended to have your 
	terminal in full screen in order to be able to fully appreciate the output) as well as storing the 
	information in a file called "adult.out". NOTE -- if there is already a file title 'adult.out', the program will 
	delete it when it begins, if you would like to make sure the correct file is affect, you can place the full 
	file path into the OUTPUT_FILE variable at the top of run.py, classifier.py, and preprocess.py.

The program stores the data from INPUT_DATASET, in this case adult-big.arff as a nested dictionary.
{ Entry#: { 'age' : age, 'education' : 'HS-grad', etc.}, Entry#: {}, etc}

At the top of preprocess.py there are a few global variables to note.
EXCLUDED_ATTRIBUTES : this list captures the attributes in the dataset that I have chosen to ignore. The 'class' is 
						ignored because I explicitly get and store the class. For reasons concerning the other attributes, 
						see the report. 
DISCRETE_TYPE : this dictionary gives the option of how to discretize the numeric categories. The system is equipped to handle
				numeric features in four different ways.
				> 1 : a number great than one signifies the user would like the attribute divided into that number of buckets
				< 1 : a number less than one signifies the user would like the discretization to term. based on this info gain threshold
				'none' : a 'none' string signifies the user would like the feature to remain numeric and for the model to use Gaussian density
				[int, int, int...] : a list of integer signifies an "expert" has predetermined the categories the feature should be divided into

				For now, only the capital-gains is discretized for reasons described in the report. However, you can feel free
				to change these values per the guidelines above and see how the results change. 

For the confusion matrixes:
	Because this is a binary classification problem, either the record is >50K or it is <=50K, there is a single
	confusion matrix produced for each run through the classifier. >50K are considered 'positive' and <=50K are
	considered 'negative'. 


Libraries Required:
time
os
math
tabulate
sys
re
operator
