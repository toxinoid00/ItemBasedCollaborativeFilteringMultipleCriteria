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
import copy
import sklearn.metrics as sm
import time

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
    query = ("SELECT user_id, hotel_id, overall_rating, value_rating, room_rating, sleep_quality_rating, location_rating, cleanliness_rating, service_rating FROM tripadvisor.reviews_2 ORDER BY user_id ASC LIMIT 100")
    df = pd.DataFrame(readFile(query),columns=["user_id","item_id","overall_rating", "value_rating", "room_rating", "sleep_quality_rating","location_rating","cleanliness_rating","service_rating"])

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

def saveToExcel(mat,name,index):
    workbook = xlsxwriter.Workbook(name+str(index)+'.xlsx')
    worksheet = workbook.add_worksheet()

    col = 0
    for row, data in enumerate(mat):
        worksheet.write_row(row, col, data)

    workbook.close()

def getRatings(df):
    ratings = []
    overall_ratings = []

    for rating in df["overall_rating"]:
        ratings.append(rating)

    overall_ratings.append(ratings)
    ratings = []

    for rating in df["value_rating"]:
        ratings.append(rating)

    overall_ratings.append(ratings)
    ratings = []

    for rating in df["room_rating"]:
        ratings.append(rating)

    overall_ratings.append(ratings)
    ratings = []

    for rating in df["sleep_quality_rating"]:
        ratings.append(rating)

    overall_ratings.append(ratings)
    ratings = []

    for rating in df["location_rating"]:
        ratings.append(rating)

    overall_ratings.append(ratings)
    ratings = []

    for rating in df["cleanliness_rating"]:
        ratings.append(rating)

    overall_ratings.append(ratings)
    ratings = []

    for rating in df["service_rating"]:
        ratings.append(rating)

    overall_ratings.append(ratings)
    ratings = []

    return overall_ratings

def readingFile():
    users = []
    items = []
    ratings = []
    df = fetchQuery()

    for user in df["user_id"]:
        users.append(user)

    for item in df["item_id"]:
        items.append(item)

    ratings = getRatings(df)

    total_users = getTotalUniqueValueArray(users)
    total_items = getTotalUniqueValueArray(items)
    total_ratings = 7
    total_df = range(len(df))
    mat_dict = []
    users_dict = getUniqueValueArray(users)
    items_dict = getUniqueValueArray(items)

    for rating_index in range(total_ratings):
        mat = createZerosMatrix(total_users,total_items)
        for index in total_df:
            user_index = getIndexArray(users_dict,users[index])
            item_index = getIndexArray(items_dict,items[index])

            mat[user_index][item_index] = ratings[rating_index][index]

        mat_dict.append(mat)

    after_random_mat_dict = []
    for rating_index in range(total_ratings):
        after_random_mat_dict.append(randomValuesWithZero(getGroundTruth(mat_dict[rating_index],total_users,total_items),0.3))

    base_mat_dict = copy.deepcopy(after_random_mat_dict[0])
    
    return after_random_mat_dict, base_mat_dict, total_users, total_items


# Similarity Measurement Modul
def similarity_item(data,total_users,total_items):
    print "\nSimilarity measurement begin..."

    items_cosine = []
    items_tanimoto = []
    items_pearson = []
    items_euclidean = []

    for rating_index in range(len(data)):
        item_cosine = createZerosMatrix(total_items,total_items)
        item_tanimoto = createZerosMatrix(total_items,total_items)
        item_pearson = createZerosMatrix(total_items,total_items)
        item_euclidean = createZerosMatrix(total_items,total_items)
        temp_item_tanimoto = createZerosMatrix(total_items,total_items)

        #Create matriks just for tanimoto only 0 and 1
        for item1 in range(total_users):
            for item2 in range(total_items):
                if data[rating_index][item1,item2] > 0:
                    temp_item_tanimoto[item1,item2] = 1.0
                else :
                    temp_item_tanimoto[item1,item2] = 0.0

        #Process similarity measurement
        for item1 in range(total_items):
            for item2 in range(total_items):
                if np.count_nonzero(data[rating_index][:,item1]) and np.count_nonzero(data[rating_index][:,item2]):
                    item_cosine[item1][item2] = 1-scipy.spatial.distance.cosine(data[rating_index][:,item1],data[rating_index][:,item2])
                    item_tanimoto[item1][item2] = scipy.spatial.distance.rogerstanimoto(temp_item_tanimoto[:,item1],temp_item_tanimoto[:,item2])
                    item_euclidean[item1][item2] = scipy.spatial.distance.euclidean(data[rating_index][:,item1],data[rating_index][:,item2])
                    try:
                        if not math.isnan(scipy.stats.pearsonr(data[rating_index][:,item1],data[rating_index][:,item2])[0]):
                            item_pearson[item1][item2] = scipy.stats.pearsonr(data[rating_index][:,item1],data[rating_index][:,item2])[0]
                        else:
                            item_pearson[item1][item2] = 0.0
                    except:
                        item_pearson[item1][item2] = 0.0
       
        items_cosine.append(item_cosine)
        items_tanimoto.append(item_tanimoto)
        items_pearson.append(item_pearson)
        items_euclidean.append(item_euclidean)

    average_similarity_cosine = average_multiple_similarity(items_cosine,total_items)
    average_similarity_tanimoto = average_multiple_similarity(items_tanimoto,total_items)
    average_similarity_pearson = average_multiple_similarity(items_pearson,total_items)
    average_similarity_euclidean = average_multiple_similarity(items_euclidean,total_items)

    sim_results = []
    sim_results.append(average_similarity_cosine)
    sim_results.append(average_similarity_tanimoto)
    sim_results.append(average_similarity_pearson)
    sim_results.append(average_similarity_euclidean)

    return sim_results

def average_multiple_similarity(data, total_items):
	result_average_similarity = createZerosMatrix(total_items,total_items)

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

    return index_rows, index_cols

def getGroundTruth(mat, total_users, total_items):
    mat_ground_truth = createZerosMatrix(total_users,total_items)
    average_users_ratings = []
    total_hotel_not_zero = 0
    for row in range(total_users):
        temp_average_user = 0.0
        total_hotel_not_zero = 0
        for col in range(total_items):
            temp_average_user = temp_average_user + mat[row][col]

            if mat[row][col] > 0 :
                total_hotel_not_zero = total_hotel_not_zero + 1

        temp_average_user = temp_average_user / total_hotel_not_zero

        temp_average_1 = temp_average_user
        temp_average_2 = temp_average_1 - int(temp_average_1)

        if temp_average_2 >= 0.5 :
            temp_average_user = math.ceil(temp_average_user)
        else :
            temp_average_user = math.floor(temp_average_user)
        
        average_users_ratings.append(temp_average_user)

    for row in range(total_users):
        for col in range(total_items):
            if mat[row][col] == 0:
                mat_ground_truth[row][col] = average_users_ratings[row]
            else :
                mat_ground_truth[row][col] = mat[row][col]

    return mat_ground_truth

def predictRating(data,base,total_users,total_items):
    print "\nProgram begin...\n"
    print "Total users: " + str(total_users)
    print "Total items: " + str(total_items)
    print "Total values: " + str(total_items*total_users)
    
    empty_index_users = []
    empty_index_items = []

    empty_index_users, empty_index_items = detectEmptyRating(data[0],total_users,total_items)
    total_empty_index = range(len(empty_index_items))
    print "Total empty value: " + str(len(total_empty_index))

    #Cosine - Tanimoto - Pearson - Euclidean
    sim_results = similarity_item(data,total_users,total_items)

    print "Similarity measurement done!"
    print "\nPredict Begin..."

    toBeRated = {"user":[], "item":[]}

    for index_row in total_empty_index:
        toBeRated["item"].append(empty_index_items[index_row])
        toBeRated["user"].append(empty_index_users[index_row])

    pred_results = []
    pred_rate = []

    for sim_index in range(len(sim_results)):
        pred_rate = []
        sim_result = sim_results[sim_index]
        for index_row in total_empty_index:
            user = toBeRated["user"][index_row]
            item = toBeRated["item"][index_row]
            
            pred = 3.0
            
            #item-based
            if np.count_nonzero(base[:,item-1]):
                sim = sim_result[item-1]
                ind = (base[user-1] > 0)
                normal = np.sum(np.absolute(sim[ind]))
                if normal > 0:
                    pred = np.dot(sim,base[user-1])/normal

            if pred < 0:
                pred = 0

            if pred > 5:
                pred = 5

            pred_rate.append(pred)

        #print "Total pred sim_index " + str(sim_index) + ": " + str(len(pred_rate))
        pred_results.append(pred_rate)

    print "\nAll predict done!\n"

    return pred_results, empty_index_users, empty_index_items, sim_results

def compareResultWithGroundTruth(base_mat, ground_truth, pred_results, total_users, total_items, empty_index_users, empty_index_items):
    mat_results = []
    mae_results = []
    for sim_index in range(len(pred_results)):
        mat_result = copy.deepcopy(base_mat)
        pred_result = pred_results[sim_index]
        for index in range(len(pred_results[0])):
            mat_result[empty_index_users[index]][empty_index_items[index]] = pred_result[index]

        mat_results.append(mat_result)
        mae_results.append(sm.mean_absolute_error(ground_truth,mat_result))

    print "MAE Cosine: " + str(mae_results[0])
    print "MAE Tanimoto: " + str(mae_results[1])
    print "MAE Pearson: " + str(mae_results[2])
    print "MAE Euclidean: " + str(mae_results[3])

    return mat_results, mae_results
    
def randomValuesWithZero(ground_truth,percent):
    temp_base_mat = copy.deepcopy(ground_truth)
    prop = int(temp_base_mat.size * percent)
    
    i = [random.choice(range(temp_base_mat.shape[0])) for _ in range(prop)]
    j = [random.choice(range(temp_base_mat.shape[1])) for _ in range(prop)]
    
    temp_base_mat[i,j] = 0.0

    return temp_base_mat

#MAIN PROGRAM
start = time.time()
temp_mat, base_mat, total_users, total_items = readingFile()
ground_truth = getGroundTruth(base_mat,total_users,total_items)
pred_results, empty_index_users, empty_index_items, sim_results = predictRating(temp_mat,base_mat,total_users,total_items)
mat_results, mae_results = compareResultWithGroundTruth(base_mat,ground_truth,pred_results,total_users,total_items, empty_index_users, empty_index_items)
end = time.time()
print "\nRuntime similarity measurement: " + str(end-start) + "s"
#END OF PROGRAM

# saveToExcel(base_mat,"base_mat",1)
# saveToExcel(ground_truth,"ground_truth",1)
# saveToExcel(sim_results[0],"sim_result",1)
# saveToExcel(sim_results[1],"sim_result",2)
# saveToExcel(sim_results[2],"sim_result",3)
# saveToExcel(sim_results[3],"sim_result",4)
# saveToExcel(mat_results[0],"mat_result",1)
# saveToExcel(mat_results[1],"mat_result",2)
# saveToExcel(mat_results[2],"mat_result",3)
# saveToExcel(mat_results[3],"mat_result",4)

# print "\nJUST FOR DEMO"
# print "\nData from users:"
# print ground_truth
# print "\nData after random 30% with zero:"
# print base_mat
# print "\nPredict using cosine"
# print mat_results[0]
# print "\nPredict using tanimoto"
# print mat_results[1]
# print "\nPredict using pearson"
# print mat_results[2]
# print "\nPredict using euclidean"
# print mat_results[3]
# print "\nMAE Cosine: " + str(mae_results[0])
# print "\nMAE Tanimoto: " + str(mae_results[1])
# print "\nMAE Pearson: " + str(mae_results[2])
# print "\nMAE Euclidean: " + str(mae_results[3])
# print "\nSimilarity matrix using cosine"
# print sim_results[0]
# print "\nSimilarity matrix using tanimoto"
# print sim_results[1]
# print "\nSimilarity matrix using pearson"
# print sim_results[2]
# print "\nSimilarity matrix using euclidean"
# print sim_results[3]
# print "\n"