#!/usr/bin/python
import mysql.connector
import pandas as pd
import numpy as np
import xlsxwriter

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
    query = ("SELECT user_id, hotel_id, overall_rating FROM tripadvisor.reviews_2 ORDER BY user_id ASC LIMIT 20")
    df = pd.DataFrame(readFile(query),columns=["user_id","hotel_id","overall_rating"])

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

# Properties
users = []
hotels = []
ratings = []
df = fetchQuery()

for user in df["user_id"]:
    users.append(user)

for hotel in df["hotel_id"]:
    hotels.append(hotel)

for rating in df["overall_rating"]:
    ratings.append(rating)

total_users = getTotalUniqueValueArray(users)
total_hotels = getTotalUniqueValueArray(hotels)
total_df = range(len(df))
mat = createZerosMatrix(total_users,total_hotels)
users_dict = getUniqueValueArray(users)
hotels_dict = getUniqueValueArray(hotels)

# Main Codes
for index in total_df:
    user_index = getIndexArray(users_dict,users[index])
    hotel_index = getIndexArray(hotels_dict,hotels[index])

    mat[user_index][hotel_index] = ratings[index]

print mat