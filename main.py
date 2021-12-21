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


# =============================================================================
# Code for Hobby?
# Which country code for Hobby most?
# Which country use open Source most?
# Male or Female code as hobby and most opensourse?
# Top programming lang on SO
# Top Desering lang acc to SO
# Top 10 Databases worked upon?
# Top desired Databases to work on
# Top 10 Platforms worked upon?
# Top desired Platforms to work on
# Top 10 Frameworks worked upon?
# Something About AI
# About Developers
# Highest Degree done by Developers
# Coders happy to code?
# =============================================================================
   
#%% correlation
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
df_encoded=pd.DataFrame(enc.fit_transform(df))

df_enc=pd.get_dummies(df)
x=enc.categories_
r = np.corrcoef(df)
#%% Questions That are not necessary or not contributing much
data=pd.read_csv('stackoverflow_data.csv')
col=data.columns
b = np.empty((98855,33))
for i in range(17,50):
    print(data[col[i]])
    data[col[i]].fillna((data[col[i]].median()), inplace=True)

#%% color bar
for  i in range(17,50):    
    b[:,i-17] = data[col[i]] 
r = np.corrcoef(b,rowvar=False)


plt.imshow(r) 
plt.colorbar()

#%% pca part
zscoredData = stats.zscore(b)
zscoredData
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_

rotatedData = pca.fit_transform(zscoredData)

covarExplained = eigVals/sum(eigVals)*100
print(covarExplained)
numClasses = 33
plt.bar(np.linspace(1,33,33),eigVals)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,numClasses],[1,1],color='red',linewidth=1) # Kaiser criterion line
print(eigVals)

#%% graphs

lst =[]
for i in range(16):
    whichPrincipalComponent = i 
    lst.append(np.argmax(np.abs(loadings[whichPrincipalComponent,:]*-1)))
    plt.figure()
    plt.bar(np.linspace(1,33,33),np.abs(loadings[whichPrincipalComponent,:]*-1))
    plt.xlabel('Question')
    plt.ylabel('Loading')
    plt.show()

#%% list of useful questions
print(lst)
questions = pd.read_csv('survey_results_schema.csv', encoding='latin-1',header=None)
for i in lst:
    print("Question =",end=" ")
    print(questions[1][i])