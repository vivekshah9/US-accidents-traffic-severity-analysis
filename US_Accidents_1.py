# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:04:14 2020

@author: vivek
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn
#Reading the US accidents data
US_accidents = pd.read_csv('C:\\ISE\\Final_Project\\US_Accidents_Dec19.csv')
US_accidents.shape
#(2974335, 49), 2974335 rows and 49 columns
#features
#49 features in the data set
#data types of my features like numeric, Categorical, etc
US_accidents.dtypes
#statistical summary of the data
summary = US_accidents.describe()
summary
head = US_accidents.head()
head
tail = US_accidents.tail()
tail

#Checking for null values
US_accidents.isnull().sum()

# Joining Number and Street name into one column 
#convert nan to 0 in Number's column
US_accidents['Number'] = US_accidents['Number'].replace(np.nan,0)

#Convert the Number column to integer

US_accidents['Number'] = US_accidents['Number'].astype(int)

#Convert the NUmber column from integer to string

US_accidents['Number'] = US_accidents['Number'].astype(str)

US_accidents['Street_Number_Name'] = US_accidents['Number'] + ' ' + US_accidents['Street']

#Drop the columns that are not necessary
US_accidents = US_accidents.drop(columns = ['ID','Source','End_Lat','End_Lng','Number','Street'])
US_accidents.info()
#Eploratory Data Analysis

#Count the number of accidents based on Severity
Severity_count = US_accidents['Severity'].value_counts()
Severity_count
Severity_level = [2,3,4,1]
Severity_levels = np.arange(len(Severity_level))
plt.bar(Severity_levels ,Severity_count)
plt.xticks(Severity_levels, Severity_level)
plt.show()

1993410 / 2974335
887620 / 2974335

sn.scatterplot(x='Start_Lng', y='Start_Lat', data= US_accidents, hue='Severity', legend='brief', s=40)
plt.show()

sn.set(style="darkgrid")

#univariate analysis to check number of accidents comparing with each variable  

#Top 15 counties with highest number of accidents. Los angeles being the county with highest accidents.
p = sn.countplot(x= US_accidents['County'], order = US_accidents['County'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

#Top 15 States with highest number of accidents. California being thr state with highest accidents.
p = sn.countplot(x= US_accidents['State'], order = US_accidents['State'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

#Top 15 Zipcodes with highest number of accidents.
p = sn.countplot(x= US_accidents['Zipcode'], order = US_accidents['Zipcode'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

#Distribution of accidents based on Timezone.
p = sn.countplot(x= US_accidents['Timezone'], order = US_accidents['Timezone'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

#Affect of Wind_Direction on accidents
p = sn.countplot(x= US_accidents['Wind_Direction'], order = US_accidents['Wind_Direction'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

#Affect of Weather_Condition on accidents
p = sn.countplot(x= US_accidents['Weather_Condition'], order = US_accidents['Weather_Condition'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

#Affect of Temperature(F) on accidents
p = sn.countplot(x= US_accidents['Temperature(F)'], order = US_accidents['Temperature(F)'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

#Affect of Humidity(%) on accidents
p = sn.countplot(x= US_accidents['Humidity(%)'], order = US_accidents['Humidity(%)'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

#Affect of Visibility(mi) on accidents
p = sn.countplot(x= US_accidents['Visibility(mi)'], order = US_accidents['Visibility(mi)'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

p = sn.countplot(x= US_accidents['Sunrise_Sunset'], order = US_accidents['Sunrise_Sunset'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

#Drop the columns Wind_Chill(F) & Precipitation(in) as it has large number of data missing
US_accidents = US_accidents.drop(columns = ['Wind_Chill(F)','Precipitation(in)'])
US_accidents.dropna(inplace = True)

#converting Start_Time and End_Time into datetime format to remove Day, Month and Year
US_accidents['Start_Time'] = pd.to_datetime(US_accidents['Start_Time'], errors='coerce')
US_accidents['End_Time'] = pd.to_datetime(US_accidents['End_Time'], errors='coerce')
US_accidents.info()

# Add Coulumns like Year, Month, Day, Hour, Weekday of when the accident took place 
US_accidents['Year']=US_accidents['Start_Time'].dt.year
US_accidents['Month']=US_accidents['Start_Time'].dt.strftime('%b')
US_accidents['Day']=US_accidents['Start_Time'].dt.day
US_accidents['Hour']=US_accidents['Start_Time'].dt.hour
US_accidents['Weekday']=US_accidents['Start_Time'].dt.strftime('%a')



p = sn.countplot(x= US_accidents['Year'], order = US_accidents['Year'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

p = sn.countplot(x= US_accidents['Month'], order = US_accidents['Month'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

#Maximum number of accidents took place from Momday to Friday 
p = sn.countplot(x= US_accidents['Weekday'], order = US_accidents['Weekday'].value_counts().iloc[0:15].index)
p.set_xticklabels(p.get_xticklabels(),rotation=45)

# Calculate the duration of traffic severity due to accidents
TD='Time_Duration(min)'
US_accidents[TD]=round((US_accidents['End_Time']-US_accidents['Start_Time'])/np.timedelta64(1,'m'))
US_accidents.info()

# Check if there is any negative time_duration values
US_accidents[TD][US_accidents[TD]<=0]

# Drop the rows with TD<0
negative_values = US_accidents[TD]<=0

# Set outliers to NAN
US_accidents[negative_values] = np.nan

# Drop rows with negative TD
US_accidents.dropna(subset=[TD],axis=0,inplace=True)
US_accidents.info()

# Double check to make sure no more negative TD
US_accidents[TD][US_accidents[TD]<=0]

#Bring the data down to state level
Cali = US_accidents[US_accidents.State == 'CA' ]

Cali = Cali.drop(columns = ['TMC','Start_Time','End_Time','Description','State','Zipcode','Country','Airport_Code','Weather_Timestamp','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight','Street_Number_Name','Year','Month','Day','Hour'])

sn.scatterplot(x='Start_Lng', y='Start_Lat', data= Cali , hue='Severity', legend='brief', s=40)
plt.show()
#Count of accidents and the severity in traffic due to accidents in CA
Severity_count = Cali['Severity'].value_counts()
Severity_count

#Bring the data down to county level. LA having highest accidents
LA = Cali[Cali.County == 'Los Angeles' ]
LA.shape
LA = LA.drop(columns = ['County'])
LA.shape

#Convert non binary categorical variables into dummy indicator variables
LA_dummy = pd.get_dummies(LA,drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# train and test split for the models
X = LA_dummy.drop(['Severity'],1)
y = np.array(LA_dummy['Severity'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=21, stratify=y)
len(X_train) #75182
len(X_test) #32221 
len(y_train) #75182
len(y_test) #32221

#Conduct LogisticRegression
clf = LogisticRegression()
model=clf.fit(X_train, y_train)
pred_test=model.predict(X_test)

target_names = ['1', '2', '3', '4']
print(classification_report(y_test, pred_test, target_names=target_names))

AS=accuracy_score(y_test, pred_test)
AS #0.8000682784519413

confusion_matrix(y_test, pred_test, labels=[1, 2, 3,4])

#Conduct K Neighbors Classifier
clf2 = KNeighborsClassifier(n_neighbors=6)
# Fit the classifier to the data
clf2.fit(X_train,y_train)
pred_test2 = clf2.predict(X_test)
AS2=accuracy_score(y_test, pred_test2)
AS2 #0.5569659538810093

#Conduct Decision Tree Classifier
clf3 = DecisionTreeClassifier(max_depth = 20 , class_weight ='balanced')
clf3
clf3.fit(X_train,y_train)
pred_test3 = clf3.predict(X_test)


AS3=accuracy_score(y_test, pred_test3)
AS3 #0.8674156605940225

clf4 = RandomForestClassifier(n_estimators = 100 ,class_weight ='balanced' )
clf4
clf4.fit(X_train,y_train)
pred_test4 = clf4.predict(X_test)
AS4=accuracy_score(y_test, pred_test3)
AS4 #0.8674156605940225




