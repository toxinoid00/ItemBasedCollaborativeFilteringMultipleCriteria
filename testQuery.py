#!/usr/bin/python
import mysql.connector
import pandas as pd
import numpy as np

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

query = ("SELECT user_id, hotel_id, overall_rating FROM tripadvisor.reviews_2 LIMIT 5")

df = pd.DataFrame(readFile(query),columns=["user_id","hotel_id","overall_rating"])

#dfPivot = df.pivot(index="user_id",columns="hotel_id",values="overall_rating")

#flattened = pd.DataFrame(dfPivot.to_records())

#writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')

#flattened.to_excel(writer, sheet_name='Sheet2')

#writer.save()