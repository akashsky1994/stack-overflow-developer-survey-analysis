# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 01:10:22 2021

@author: mohit
"""
   
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats 
from sklearn.decomposition import PCA
import prince

   
#%% 


df=pd.read_csv('stackoverflow_data.csv')
col=df.columns

df.isnull().sum()
df.info()
df.describe()

df.fillna(method='ffill', inplace=True)
df.isnull().sum()
df['MilitaryUS'].fillna('No', inplace=True)
df.isnull().sum()
#Respodents group by country

   
#%% Hypothesis US and Indian Developers are equally satisfied by their job.

df_country= (df.groupby(['Country'])).count()
df_country.sort_values(by=['Respondent'], ascending=False, inplace=True)

df_country.iloc[0:10, 0].plot.bar()


USPeople_df= df[df['Country']== 'United States']
IndianPeople_df= df[df['Country']== 'India']
USPeople_df=USPeople_df[['Country','JobSatisfaction']]
IndianPeople_df=IndianPeople_df[['Country','JobSatisfaction']]
#decide hypothesis Akash
   
#%% Hypothesis for US male and female developers equally paid
#Code is designed in such way can compare for any country.


df_allgender=df.groupby(['Gender']).count()
df_allgender.sort_values(by=['Respondent'], ascending=False, inplace=True)
df_allgender.iloc[0:3, 0].plot.bar()

df_allgender=df_allgender.T
df_gender= df_allgender[['Female', 'Male']]
df_gender=df_gender.iloc[0, :]

df_female= df[df['Gender']== 'Female']
df_male= df[df['Gender']== 'Male']
femaleSalaries_df= df_female[['Country','Currency','CurrencySymbol','Salary']]
maleSalaries_df= df_male[['Country','Currency','CurrencySymbol','Salary']]


USFemaleSalaries=pd.DataFrame(femaleSalaries_df.groupby('Currency')['Salary'])
USMaleSalaries=pd.DataFrame(maleSalaries_df.groupby('Currency')['Salary'])
M1=USFemaleSalaries.iloc[18, 1]
M2=USMaleSalaries.iloc[18,1]

u,p1 = stats.mannwhitneyu(M1,M2)
k,p=stats.kstest(M1,M2)
   
#%% Same for Student and Hobby/Salary/Race Ethinity are Student with anythihng else, think about it
#Use of open Source compare to Country/Gender
#Education and Salary comparison

 
#%% Other Ideas


codeforHobby_df= df[df['Hobby']== 'Yes']

# Which country code for Hobby most?
codeforHobbyCountry_df= codeforHobby_df.groupby('Country').count()
codeforHobbyCountry_df.sort_values(by=['Respondent'], ascending=False, inplace=True)

# What Age group codes for Hobby most?
codeforHobbyAge_df= codeforHobby_df.groupby('Age').count()
codeforHobbyAge_df.sort_values(by=['Respondent'], ascending=False, inplace=True)

# How many years are developer coding for Hobby most?
codeforHobbyYearsCoding_df= codeforHobby_df.groupby('YearsCoding').count()
codeforHobbyYearsCoding_df.sort_values(by=['Respondent'], ascending=False, inplace=True)

codeforHobbyCountry_df.iloc[0:10, 0].plot.bar()
codeforHobbyAge_df.iloc[:, 0].plot.bar()
codeforHobbyYearsCoding_df.iloc[:, 0].plot.bar()

#%% Which country use open Source most?
# Male or Female code as hobby and most opensourse?

openSource_df= df[df['OpenSource']== 'Yes']

openSourceCountry_df= openSource_df.groupby('Country').count()
openSourceCountry_df.sort_values(by=['Respondent'], ascending=False, inplace=True)

openSourceCountry_df.iloc[0:10, 0].plot.bar()

openSource_df=pd.DataFrame([df[(df['Gender'] == 'Female') & (df['OpenSource'] == 'Yes')].count()[0], df[(df['Gender'] == 'Male') & (df['OpenSource'] == 'Yes')].count()[0]])
openSource_df=openSource_df.T
openSource_df.columns=['Female', 'Male']
openSource_df.plot.bar()


#%% 
# Top 10 Databases worked upon?

databases_df= df['DatabaseWorkedWith'].value_counts()
databases_df.iloc[0:5].plot.bar()

# Top desired Databases to work on

desiredDatabase_df=df['DatabaseDesireNextYear'].value_counts()
desiredDatabase_df.iloc[0:5].plot.bar()

#%% 
# Top 10 Platforms worked upon?
# Top desired Platforms to work on
# Top 10 Frameworks worked upon?

platform_df= df['PlatformWorkedWith'].value_counts()
platform_df.iloc[0:5].plot.bar()

# Top desired Databases to work on

desiredplatform_df=df['PlatformDesireNextYear'].value_counts()
desiredplatform_df.iloc[0:5].plot.bar()

framework_df= df['FrameworkWorkedWith'].value_counts()
framework_df.iloc[0:10].plot.bar()

#%% 
# About Developer

# Coders happy to code?
satisfaction_df= df['JobSatisfaction'].value_counts()
satisfaction_df.plot.pie()

#Most Common Jobs for Developers
job_df=df['DevType'].value_counts()
job_df.iloc[0:10].plot.pie()

# Highest Degree done by Developers
degree_df= df['FormalEducation'].value_counts()
degree_df.plot.pie()


#%% correlation
from sklearn.preprocessing import OneHotEncoder
#Find the unique in every column and determine which one to OnehOTEncode
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df.values)
x=enc.transform(df.values)
#Select Which columns to use for prediction
#Add it to rest columns and apply PCA
#And Than predict


df_enc=pd.get_dummies(df)
x=enc.categories_
r = np.corrcoef(df)
