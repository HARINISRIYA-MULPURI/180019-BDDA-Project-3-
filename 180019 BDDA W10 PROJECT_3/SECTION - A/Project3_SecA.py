#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[43]:


# importing datastets
titanic_data=pd.read_csv(r"G:\semester 7\Bigdata-2\Week 9\Titanic.csv")
test_data=pd.read_csv(r"G:\semester 7\Bigdata-2\Week 9\test.csv")


# In[44]:


titanic_data.head


# In[47]:


titanic_data.describe


# In[48]:


print("Missing values in the data values")
total = titanic_data.isnull().sum().sort_values(ascending=False)
percent_1 = titanic_data.isnull().sum()/titanic_data.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(missing_data)


# #### Ques 1.) Find out the overall chance of survival for a Titanic passenger.
# 

# In[49]:


print("Total number of passengers survived are : ",titanic_data['survived'].value_counts()[1])
print("Percentage passengers survived are : ",titanic_data['survived'].value_counts(normalize=True)[1]*100)


# #### Ques 2.) Find out the chance of survival for a Titanic passenger based on their sex and plot it.

# In[50]:


sns.barplot(x="sex", y="survived", data=titanic_data)
print("The percentage of females who survived are : ", titanic_data["survived"][titanic_data["sex"] == 'female'].value_counts(normalize = True)[1]*100)
print("The percentage of males who survived are : ", titanic_data["survived"][titanic_data["sex"] == 'male'].value_counts(normalize = True)[1]*100)


# #### Ques 3.) Find out the chance of survival for a Titanic passenger by traveling class wise and plot it.

# In[51]:


sns.barplot(x="pclass", y="survived", data=titanic_data)
print("The percentage of Pclass 1 who survived are  : ", titanic_data["survived"][titanic_data["pclass"] == 1].value_counts(normalize = True)[1]*100)
print("The percentage of Pclass 2 who survived are : ", titanic_data["survived"][titanic_data["pclass"] == 2].value_counts(normalize = True)[1]*100)
print("The percentage of Pclass 3 who survived are : ", titanic_data["survived"][titanic_data["pclass"] == 3].value_counts(normalize = True)[1]*100)


# #### Ques 4.) Find out the average age for a Titanic passenger who survived by passenger class and sex. 

# In[52]:


fig = plt.figure(figsize=(12,5))
fig.add_subplot(121)
plt.title('Survivors Age/Sex per Passenger Class')
sns.barplot(data=titanic_data, x='pclass',y='age',hue='sex')


# In[53]:


meanAgeTrnMale = round(titanic_data[(titanic_data['sex'] == "male")]['age'].groupby(titanic_data['pclass']).mean(),2)
meanAgeTrnFeMale = round(titanic_data[(titanic_data['sex'] == "female")]['age'].groupby(titanic_data['pclass']).mean(),2)


print('Mean age per sex per pclass')
print(pd.concat([meanAgeTrnMale, meanAgeTrnFeMale], axis = 1,keys= ['Male','Female']))


# #### Ques 5.) Find out the chance of survival for a Titanic passenger based on number of siblings the passenger had on the ship and plot it.

# In[54]:


sns.barplot(x="sibsp", y="survived", data=titanic_data)
plt.title('Passangers Survival chance based on number of siblings the passenger')
print("The percentage of SibSp 0 who survived are : ", titanic_data["survived"][titanic_data["sibsp"] == 0].value_counts(normalize = True)[1]*100)
print("The percentage of SibSp 1 who survived are : ", titanic_data["survived"][titanic_data["sibsp"] == 1].value_counts(normalize = True)[1]*100)
print("The percentage of SibSp 2 who survived are : ", titanic_data["survived"][titanic_data["sibsp"] == 2].value_counts(normalize = True)[1]*100)


# #### Ques 6.) Find out the chance of survival for a Titanic passenger based on number of parents/children the passenger had on the ship and plot it.

# In[55]:


sns.barplot(x="parch", y="survived", data=titanic_data)
plt.title('Passangers Survival chance based on number of parents/childrens')
plt.show()
print("The percentage of parch 0 who survived are : ", titanic_data["survived"][titanic_data["parch"] == 0].value_counts(normalize = True)[1]*100)
print("The Percentage of parch 1 who survived are : ", titanic_data["survived"][titanic_data["parch"] == 1].value_counts(normalize = True)[1]*100)
print("The Percentage of parch 2 who survived are : ", titanic_data["survived"][titanic_data["parch"] == 2].value_counts(normalize = True)[1]*100)
print("The Percentage of parch 3 who survived are : ", titanic_data["survived"][titanic_data["parch"] == 3].value_counts(normalize = True)[1]*100)


# #### Ques 7.) Plot out the variation of survival and death amongst passengers of different age.

# In[56]:


titanic_data["age"] = titanic_data["age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
titanic_data['agegroup'] = pd.cut(titanic_data['age'], bins, labels = labels)
sns.barplot(x="agegroup", y="survived", data=titanic_data)
plt.title('variation of survival and death amongst passengers of different age')
plt.show()


# In[57]:


g = sns.FacetGrid(titanic_data, col='survived')
g.map(plt.hist, 'age', bins=20)


# #### Ques 8.) Plot out the variation of survival and death with age amongst passengers of different passenger classes.

# In[58]:


print("variation of survival and death with age and class")
grid = sns.FacetGrid(titanic_data, col='survived', row='pclass', size=3, aspect=2)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();


# #### Ques 9.) Find out the survival probability for a Titanic passenger based on title from the name of passenger.

# In[59]:


combine = [titanic_data, test_data]
for dataset in combine:
    dataset['Title'] = dataset.name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(titanic_data['Title'],titanic_data['sex'])


# In[60]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

titanic_data[['Title', 'survived']].groupby(['Title'], as_index=False).mean()


# #### Ques 10.) What conclusions are you derived from the analysis?
# ##### Written in the document.

# In[ ]:




