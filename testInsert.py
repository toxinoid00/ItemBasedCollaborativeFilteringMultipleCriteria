#!/usr/bin/python
import mysql.connector

cnx = mysql.connector.connect(user='test', host='127.0.0.2', password='Husen123', database='tripadvisor2')
cursor = cnx.cursor()

insert_review = ("INSERT INTO testing "
               "(reviews_id, customer_id, hotel_id, rating_0) "
               "VALUES (%s, %s, %s, %s)")

data = ('13','4','14','4.0')

cursor.execute(insert_review,data)

cnx.commit()

cursor.close()
cnx.close()