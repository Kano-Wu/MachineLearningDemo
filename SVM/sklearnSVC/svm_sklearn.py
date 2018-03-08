import csv
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import math

np.random.seed(1024)

def load_iris_data(data_file, shuffle=True):
	X = []
	Y = []

	lines = csv.reader(open(data_file, 'rb'))
	dataset = list(lines)
	for i in range(len(dataset)):
		X.append([float(x) for x in dataset[i][:-1]]) 
		Y.append(int(dataset[i][-1]))

	if shuffle:
		Data = zip(X, Y)
		np.random.shuffle(Data)
		X, Y = zip(*Data)

	X = np.array(X)
	Y = np.array(Y)
	return [X, Y]

def train_svm_by_sklearnSVC(Xtrain, ytrain):
	# svm
	clf = SVC()
	# train
	clf.fit(Xtrain, ytrain)
	return clf

def predict_by_sklearnSVC(clf, Xtest):
	y_pred = clf.predict(Xtest)
	return y_pred

def calculate_metrics(y_true, y_pred):
	# Report
	test_classify_report = metrics.classification_report(y_true, y_pred)
	# Accuracy
	test_acc = metrics.accuracy_score(y_true, y_pred)
	# Macro F1-Score
	test_f1 = metrics.f1_score(y_true, y_pred, average="macro")
	# MSE : mean squared error
	test_mse = metrics.mean_squared_error(y_true, y_pred)
	# RMSE : root mean square error
	test_rmse = math.sqrt(test_mse)

	# Output
	print test_classify_report
	print 
	print 'Acc: ', test_acc
	print 'F1:  ', test_f1
	print 'MSE: ', test_mse
	print 'RMSE:', test_rmse

if __name__ == '__main__':
	data_dir = '../../data/'

	iris_file = data_dir + 'iris.csv'
	pima_file = data_dir + 'pima.csv'

	X, Y = load_iris_data(iris_file)
	print 'data loaded.'

	dataset_size = len(X)
	val_split = 0.2
	val_size = int(dataset_size*val_split)

	# split
	Xtrain = X[:val_size]
	ytrain = Y[:val_size]
	Xtest = X[val_size:]
	ytest = Y[val_size:]

	print 'train size: ', len(Xtrain)
	print 'test  size: ', len(Xtest)

	# train 
	clf = train_svm_by_sklearnSVC(Xtrain, ytrain)
	# predict
	y_pred = predict_by_sklearnSVC(clf, Xtest)

	# calculate metrics
	calculate_metrics(ytest, y_pred)
