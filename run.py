import preprocess
import classifier
import time
import os


INPUT_DATASET = 'adult-big.arff'
OUTPUT_FILE = 'adult.out'

if __name__ == "__main__":
	"""
	Main method which controls
	the data input and preprocessing,
	and runs 10-fold cross-validaton
	on a Naive Bayes classification model.

	Output to screen will provide
	information on the program as it runs.

	"""

	try:
		os.remove(OUTPUT_FILE)
	except OSError:
		pass

	with open(OUTPUT_FILE, "w") as out:
		out.write('\n\nStarting Time: ')
		out.write(time.asctime(time.localtime(time.time())))
		out.write('\n\n')

	print "Please expect approximately 1 minute and 15 seconds for the classification to complete."

	data, features, totals = preprocess.control(INPUT_DATASET)

	classifier.run_10_fold(data, totals, features)

	with open(OUTPUT_FILE, "a") as out:
		out.write('\n\nEnding Time: ')
		out.write(time.asctime(time.localtime(time.time())))
		out.write('\n\n')
