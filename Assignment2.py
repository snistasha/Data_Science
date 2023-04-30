#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import glob

import zipfile
import csv
import warnings
warnings.filterwarnings('ignore')
import gc
import datetime


# In[2]:


survey_data = pd.read_excel('SurveyResults.xlsx',usecols=['ID', 'Start time', 'End time', 'date', 'Stress level'])


# In[3]:


survey_data.dtypes


# In[4]:


survey_data.isnull().sum()


# In[5]:


survey_data['Start datetime'] =  pd.to_datetime(survey_data['date'].map(str) + ' ' + survey_data['Start time'].map(str))
survey_data['End datetime'] =  pd.to_datetime(survey_data['date'].map(str) + ' ' + survey_data['End time'].map(str))
survey_data.drop(['Start time', 'End time', 'date'], axis=1, inplace=True)

survey_data['datetime'] = survey_data['End datetime'] - survey_data['Start datetime']
survey_data['time'] = survey_data['datetime'].apply(lambda x: x.seconds/60)


# In[7]:


survey_data.head()


# In[81]:


final_df = pd.DataFrame()
count = 0
print('There are total',len(glob.glob(os.path.join(r"C:\Users\shant\Downloads\2201038\Data\*\*"))),'zipfiles files in the dataset')
for i in glob.glob(os.path.join(r"C:\Users\shant\Downloads\2201038\Data\*\*")):
    with zipfile.ZipFile(i, 'r') as zip_ref:
        
        ACC = pd.read_csv(zip_ref.open('ACC.csv'))
        BVP = pd.read_csv(zip_ref.open('BVP.csv'))
        EDA = pd.read_csv(zip_ref.open('EDA.csv'))
        HR = pd.read_csv(zip_ref.open('HR.csv'))
        TEMP = pd.read_csv(zip_ref.open('TEMP.csv'))
        
        ACC['time'] = int(float(ACC.columns[0]))
        ACC['time'] = ACC['time']+ACC.index
        ACC['time'] = ACC['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
        
        def process_df(df):
            start_timestamp = df.iloc[0,0]
            sample_rate = df.iloc[1,0]
            new_df = pd.DataFrame(df.iloc[2:].values, columns=df.columns)
            new_df['datetime'] = [(start_timestamp + i/sample_rate) for i in range(len(new_df))]
            return new_df
        
        ACC = process_df(ACC)
        BVP = process_df(BVP)
        EDA = process_df(EDA)
        HR = process_df(HR)
        TEMP = process_df(TEMP)
        
        ACC.rename({ACC.columns[0]:'X',ACC.columns[1]:'Y',ACC.columns[2]:'Z',},axis = 1,inplace = True)
        BVP.rename({BVP.columns[0]:'BVP'},axis = 1,inplace = True)
        EDA.rename({EDA.columns[0]:'EDA'},axis = 1,inplace = True)
        HR.rename({HR.columns[0]:'HR'},axis = 1,inplace = True)
        TEMP.rename({TEMP.columns[0]:'TEMP'},axis = 1,inplace = True)
        
        final = ACC.merge(BVP,on = 'datetime',how = 'outer').merge(EDA,on = 'datetime',how = 'outer').merge(HR,on = 'datetime',how = 'outer').merge(TEMP,on = 'datetime',how = 'outer')
        
        final = final.fillna(method='ffill')
        final = final.fillna(method='bfill').reset_index(drop = True)
        
        final['Stress'] = np.where(final['datetime'] > survey_data['time'].max(),2,1)
        
        final.drop_duplicates(inplace = True)
        
        final_df = final_df.append(final)
        
        count += 1
            
        gc.collect()
        
final_df.drop_duplicates(inplace = True)
final_df.to_csv('final.csv',index = False)
print('Preprocessing is done')


# # Assignment 2

# In[43]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[44]:


#Reading the Final CSV
final = pd.read_csv('final.csv',parse_dates = ['time'])


# In[45]:


final.head()


# In[46]:


#Checking Null values
final.isnull().sum()


# In[47]:


#checking duplicates
final.duplicated().sum()


# In[48]:


final.info()


# In[49]:


final.dtypes


# In[50]:


final['Stress'].value_counts()


# In[51]:


sns.countplot(final['Stress'])


# # Handling the imbalance data

# In[52]:


# Separate majority and minority classes
majority_class = final[final.Stress == 1]
minority_class = final[final.Stress == 2]

downsampled_majority_class = resample(majority_class,replace=False,n_samples=len(minority_class))
final = pd.concat([downsampled_majority_class, minority_class]).sample(frac=1).reset_index(drop = True)


# In[53]:


final


# In[54]:


final.Stress.value_counts()


# In[55]:


sns.countplot(final['Stress'])


# # Correlation to determine which feature is good

# In[56]:


final.corr()


# In[57]:


sns.set(rc={"figure.figsize":(10,10)})
sns.heatmap(final.corr(),annot = True)


# # Time series

# In[72]:


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


# In[83]:


fig, ax = plt.subplots(figsize=(20, 20))
ax.plot(final.time, final.X)
ax.plot(final.time, final.Y)
ax.plot(final.time, final.Z)
ax.set_title('ACC')


# In[82]:


fig, ax = plt.subplots(figsize=(20, 20))
ax.plot(final.time, final.HR)
ax.set_title('HEART RATE')


# # Splitting the data

# In[61]:


final.columns


# In[62]:


X = final.drop({'time','Stress','datetime'},axis = 1)
y = final['Stress']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# # Model Building

# In[63]:


#Logistic Regression

LR = LogisticRegression()
LR.fit(X_train, y_train)


# In[64]:


model1 = LR.predict(X_test)


# In[65]:


print(classification_report(y_test, model1))


# In[ ]:





# In[66]:


#Naive Bayes

GNB = GaussianNB()
GNB.fit(X_train, y_train)


# In[67]:


model2 = GNB.predict(X_test)


# In[68]:


print(classification_report(y_test, model2))


# In[ ]:





# In[69]:


#Random Forest

RF = RandomForestClassifier()
RF.fit(X_train, y_train)


# In[70]:


model3= RF.predict(X_test)


# In[71]:


print(classification_report(y_test, model3))


# In[ ]:




