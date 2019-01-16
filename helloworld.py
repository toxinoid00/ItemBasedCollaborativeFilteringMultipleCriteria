import numpy as np
import scipy.stats
import scipy.spatial
import random
from math import sqrt
import math
import warnings
import sys
from sklearn.model_selection import train_test_split

#Total items and users
items = 4
users = 4
total_rating = 3

def readingFile(filename):
	f = open(filename,"r")
	data = []
	datas = []

	#append all file data
	for row in f:
		r_split = row.split(',')
		e = []
		for coloumn in range(len(r_split)):
			e.append(int(r_split[coloumn]))
		data.append(e)

	#split data every category rating
	for rating_index in range(total_rating):
		temp_data = []
		for row in data:
			temp_data.append([row[0],row[1],row[rating_index+2]])

		datas.append(temp_data)
	return datas

# Similarity Measurement Modul
def similarity_item(data):
	print "Similarity Measurement Begin..."
	
	items_cosine = []
	items_tanimoto = []
	items_pearson = []
	items_euclidean = []

	for rating_index in range(len(data)):
		item_cosine = np.zeros((items,items))
		item_tanimoto = np.zeros((items,items))
		item_pearson = np.zeros((items,items))
		item_euclidean = np.zeros((items,items))
		temp_item_tanimoto = np.zeros((items,items)) 

		for item1 in range(items):
			for item2 in range(items):
				if data[rating_index][item1,item2] > 0:
					temp_item_tanimoto[item1][item2] = 1.0
				else:
					temp_item_tanimoto[item1][item2] = 0.0

		for item1 in range(items):
			for item2 in range(items):
				item_cosine[item1][item2] = 1-scipy.spatial.distance.cosine(data[rating_index][:,item1],data[rating_index][:,item2])
				item_tanimoto[item1][item2] = scipy.spatial.distance.rogerstanimoto(temp_item_tanimoto[:,item1],temp_item_tanimoto[:,item2])
				item_pearson[item1][item2] = scipy.stats.pearsonr(data[rating_index][:,item1],data[rating_index][:,item2])[0]
				item_euclidean[item1][item2] = scipy.spatial.distance.euclidean(data[rating_index][:,item1],data[rating_index][:,item2])

		items_cosine.append(item_cosine)
		items_tanimoto.append(item_tanimoto)
		items_pearson.append(item_pearson)
		items_euclidean.append(item_euclidean)

	average_similarity_cosine = average_multiple_similarity(items_cosine)
	average_similarity_tanimoto = average_multiple_similarity(items_tanimoto)
	average_similarity_pearson = average_multiple_similarity(items_pearson)
	average_similarity_euclidean = average_multiple_similarity(items_euclidean)

	#print average_similarity_cosine
	#print average_similarity_tanimoto
	#print average_similarity_pearson
	#print average_similarity_euclidean

	return average_similarity_cosine, average_similarity_tanimoto, average_similarity_pearson, average_similarity_euclidean
    
def average_multiple_similarity(data):
	result_average_similarity = np.zeros((items,items))

	for item1 in range(items):
		for item2 in range(items):
			temp_average = 0.0
			for rating_index in range(len(data)):
				temp_average = temp_average + data[rating_index][item1,item2]
			
			temp_average = temp_average / len(data)
			result_average_similarity[item1][item2] = temp_average

	return result_average_similarity

def detectEmptyRating(data):
	index_rows = []
	index_cols = []

	for row in range(items):
		for col in range(items):
			if data[0][row,col] == 0:
				index_cols.append(col)
				index_rows.append(row)

	return index_cols, index_rows

def predictRating(recommend_data):
	Mat = []

	#train, test = train_test_split(recommend_data[0], test_size=0.3)

	for a in recommend_data:
		temp_mat = np.zeros((users,items))
		for e in a:
			temp_mat[e[0]-1][e[1]-1] = e[2]

		Mat.append(temp_mat)

	sim_cosine_item, sim_tanimoto_item, sim_pearson_item, sim_euclidean_item = similarity_item(Mat)

	index_cols, index_rows = detectEmptyRating(Mat)

	f = open(sys.argv[2],"r")
	toBeRated = {"user":[], "item":[]}
	for row in f:
		r = row.split(',')	
		toBeRated["item"].append(int(r[1]))
		toBeRated["user"].append(int(r[0]))

	f.close()

	pred_rate = []

	fw_w = open('results.csv','w')

	l = len(toBeRated["user"])
	for e in range(l):
		user = toBeRated["user"][e]
		item = toBeRated["item"][e]

		pred = 3.0

		#item-based
		if np.count_nonzero(Mat[0][:,item-1]):
			sim = sim_cosine_item[item-1]
			ind = (Mat[0][user-1] > 0)
			#ind[item-1] = False
			normal = np.sum(np.absolute(sim[ind]))
			if normal > 0:
				pred = np.dot(sim,Mat[0][user-1])/normal

		if pred < 0:
			pred = 0

		if pred > 5:
			pred = 5

		pred_rate.append(pred)
		print str(user) + "," + str(item) + "," + str(pred)
		#fw.write(str(user) + "," + str(item) + "," + str(pred) + "\n")
		fw_w.write(str(pred) + "\n")

	#fw.close()
	fw_w.close()

recommend_data  = readingFile(sys.argv[1])

predictRating(recommend_data)