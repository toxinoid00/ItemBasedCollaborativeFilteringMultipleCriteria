#!/usr/bin/python
import mysql.connector
import pandas as pd
import numpy as np
import xlsxwriter
import scipy.stats
import scipy.spatial
import random
from math import sqrt
import math
import warnings
import sys

# Functions
def connectToMySQL(user,host,password,database):
    connection = mysql.connector.connect(user=user, 
                                        host=host, 
                                        password=password, 
                                        database=database)

    return connection

def readFile(query):
    connection = connectToMySQL("root","127.0.0.1","husen123","tripadvisor")
    cursor = connection.cursor()

    cursor.execute(query)

    result = cursor.fetchall()

    cursor.close()
    connection.close()

    return result

def getTotalUniqueValueArray(array) :
    return len(set(array))

def fetchQuery():
    query = ("SELECT user_id, hotel_id, overall_rating FROM tripadvisor.reviews_2 ORDER BY user_id ASC LIMIT 1000")
    df = pd.DataFrame(readFile(query),columns=["user_id","item_id","overall_rating"])

    return df

def getIndexArray(array, value):
    return array.index(value)

def getUniqueValueArray(array):
    tmp_array_dict = []
    for tmp_arr in array:
        if tmp_arr not in tmp_array_dict :
            tmp_array_dict.append(tmp_arr)

    return tmp_array_dict

def createZerosMatrix(row,col):
    return np.zeros((row,col))

def saveToExcel(mat):
    workbook = xlsxwriter.Workbook('testExcel.xlsx')
    worksheet = workbook.add_worksheet()

    col = 0
    for row, data in enumerate(mat):
        worksheet.write_row(row, col, data)

    workbook.close()

def readingFile():
    users = []
    items = []
    ratings = []
    df = fetchQuery()

    for user in df["user_id"]:
        users.append(user)

    for item in df["item_id"]:
        items.append(item)

    for rating in df["overall_rating"]:
        ratings.append(rating)

    total_users = getTotalUniqueValueArray(users)
    total_items = getTotalUniqueValueArray(items)
    total_df = range(len(df))
    mat = createZerosMatrix(total_users,total_items)
    users_dict = getUniqueValueArray(users)
    items_dict = getUniqueValueArray(items)

    for index in total_df:
        user_index = getIndexArray(users_dict,users[index])
        item_index = getIndexArray(items_dict,items[index])

        mat[user_index][item_index] = ratings[index]

    return mat, total_users, total_items


# Similarity Measurement Modul
def similarity_item(data,total_items):
	print "Similarity Measurement Begin..."
	item_similarity_cosine = np.zeros((total_items,total_items))

	for item1 in range(total_users):
		for item2 in range(total_items):
			item_similarity_cosine[item1][item2] = 1-scipy.spatial.distance.cosine(data[:,item1],data[:,item2])	

	return item_similarity_cosine
    
def average_multiple_similarity(data, total_items):
	result_average_similarity = np.zeros((total_items,total_items))

	for item1 in range(total_items):
		for item2 in range(total_items):
			temp_average = 0.0
			for rating_index in range(len(data)):
				temp_average = temp_average + data[rating_index][item1,item2]
			
			temp_average = temp_average / len(data)
			result_average_similarity[item1][item2] = temp_average

	return result_average_similarity

def detectEmptyRating(data,total_users,total_items):
    index_rows = []
    index_cols = []
    
    for row in range(total_users):
		for col in range(total_items):
			if data[row][col] == 0:
				index_cols.append(col)
				index_rows.append(row)

    total_index = range(len(index_cols))
    return index_rows, index_cols, total_index

def predictRating(data,total_users,total_items):
    sim_cosine_item = similarity_item(data,total_items)

    empty_index_users, empty_index_items, total_index = detectEmptyRating(data,total_users,total_items)

    toBeRated = {"user":[], "item":[]}
    for index_row in total_index:
		toBeRated["item"].append(empty_index_items[index_row])
		toBeRated["user"].append(empty_index_users[index_row])

    pred_rate = []
    
    for index_row in total_index:
		user = toBeRated["user"][index_row]
		item = toBeRated["item"][index_row]

		pred = 3.0

		#item-based
		if np.count_nonzero(data[:,item-1]):
			sim = sim_cosine_item[item-1]
			ind = (data[user-1] > 0)
			#ind[item-1] = False
			normal = np.sum(np.absolute(sim[ind]))
			if normal > 0:
				pred = np.dot(sim,data[user-1])/normal

		if pred < 0:
			pred = 0

		if pred > 5:
			pred = 5

		pred_rate.append(pred)

    print pred_rate


temp_mat, total_users, total_items = readingFile()
predictRating(temp_mat,total_users,total_items)
