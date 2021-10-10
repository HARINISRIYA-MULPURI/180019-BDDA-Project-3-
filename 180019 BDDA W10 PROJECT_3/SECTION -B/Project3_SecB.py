#!/usr/bin/env python
# coding: utf-8

# In[109]:



import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[180]:


train_data = pd.read_csv(r"G:\semester 7\Bigdata-2\Week 9\Titanic.csv") # we have already trained titanic data set
test_data = pd.read_csv(r"G:\semester 7\Bigdata-2\Week 9\test.csv")
combine = [train_data, test_data]
combine


# In[111]:


train_data.head()


# In[112]:


train_data.tail()


# # Data Analysis

# In[113]:


train_data.info()
print('_'*40)
test_data.info()


# In[114]:


train_data.describe()


# In[115]:


train_data.describe(include=['O'])


# # Analyzing data by pivot table

# In[116]:


train_data[['pclass','survived']].groupby(['pclass'],as_index=False).mean()


# In[117]:


train_data[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[118]:


train_data[["sex", "survived"]].groupby(['sex'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[119]:


train_data[["sibsp", "survived"]].groupby(['sibsp'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[120]:


train_data[["parch", "survived"]].groupby(['parch'], as_index=False).mean().sort_values(by='parch',ascending=True)


# In[121]:


train_data[['fare']].describe()


# # Data Analysis by visualising

# In[122]:


g = sns.FacetGrid(train_data, col='pclass')
g.map(plt.hist,'fare', bins=10)


# In[123]:


g = sns.FacetGrid(train_data, col='survived')
g.map(plt.hist, 'age', bins=20)


# In[124]:


grid = sns.FacetGrid(train_data, col='survived', row='pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();


# In[128]:



grid = sns.FacetGrid(train_data, row='embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'pclass', 'survived','sex', palette='deep')
grid.add_legend()


# In[129]:


grid = sns.FacetGrid(train_data, row='sex', col='survived', size=2.2, aspect=1.6)
grid.map(plt.hist,'age',bins=20)
grid.add_legend();


# In[130]:


grid = sns.FacetGrid(train_data, row='embarked', col='survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'sex', 'fare', alpha=.5, ci=None)
grid.add_legend()


# # Data wrangling

# In[138]:


print("Before", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)

train_data = train_data.drop(['ticket', 'cabin'], axis=1)
test_data = test_data.drop(['ticket', 'cabin'], axis=1)
combine = [train_data, test_data]

"After", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape


# # Extracting from existing

# In[140]:


for dataset in combine:
    dataset['Title'] = dataset.name.str.extract(' ([A-Za-z]+)\.', expand=False)


pd.crosstab(train_data['Title'], train_data['sex'])


# In[141]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_data[['Title', 'survived']].groupby(['Title'], as_index=False).mean()


# In[142]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_data.head()


# In[143]:


train_data['Title'].value_counts()


# In[144]:


print(combine[0].head())


# In[145]:


train_data = train_data.drop(['name', 'passengerId'], axis=1)
test_data = test_data.drop(['name'], axis=1)
combine = [train_data, test_data]
train_data.shape, test_data.shape


# In[146]:


train_data.shape
train_data.head()


# # Converting a categorical feature

# In[147]:


for dataset in combine:
    dataset['sex'] = dataset['sex'].map( {'female': 1, 'male': 0}).astype(int)

print(train_data.head())


# In[148]:


print (test_data.head())


# # Numerical continuous feature

# In[149]:


grid = sns.FacetGrid(train_data, row='pclass', col='sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend()


# In[150]:


guess_ages = np.zeros((2,3))
guess_ages


# In[151]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['sex'] == i) & (dataset['pclass'] == j+1)]['age'].dropna()

            age_guess = guess_df.median()

           
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.age.isnull()) & (dataset.sex == i) & (dataset.pclass == j+1),'age'] = guess_ages[i,j]

    dataset['age'] = dataset['age'].astype(int)

train_data.head()


# In[152]:


train_data['AgeBand'] = pd.cut(train_data['age'], 5)
train_data[['AgeBand', 'survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[153]:


for dataset in combine:    
    dataset.loc[ dataset['age'] <= 16, 'age'] = 0
    dataset.loc[(dataset['age'] > 16) & (dataset['age'] <= 32), 'age'] = 1
    dataset.loc[(dataset['age'] > 32) & (dataset['age'] <= 48), 'age'] = 2
    dataset.loc[(dataset['age'] > 48) & (dataset['age'] <= 64), 'age'] = 3
    dataset.loc[ dataset['age'] > 64, 'age']
train_data.head()


# In[154]:


train_data = train_data.drop(['AgeBand'], axis=1)
combine = [train_data, test_data]
train_data.head()
test_data.head()


# In[52]:


train_data.head()


# In[155]:


test_data.head()


# # Combining existing features

# In[156]:


for dataset in combine:
    dataset['FamilySize'] = dataset['sibsp'] + dataset['parch'] + 1


# In[157]:


train_data.head()
train_data[['FamilySize', 'survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[158]:


train_data.info()


# In[159]:


train_data['FamilySize'].value_counts()


# In[160]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


# In[161]:


train_data.head()
train_data[['IsAlone', 'survived']].groupby(['IsAlone'], as_index=False).mean()


# In[162]:


dropped_one = train_data['parch']
dropped_two = train_data['sibsp']
dropped_three = train_data['FamilySize']
dropped_one


# In[163]:


test_data.head()


# In[164]:


combine = [train_data, test_data]

train_data.head()


# In[165]:


for dataset in combine:
    dataset['age*Class'] = dataset.age * dataset.pclass

train_data.loc[:, ['age*Class', 'age', 'pclass']].head(10)


# In[166]:


train_data['age*Class'].value_counts()


# # Completing categorical feature

# In[182]:


freq_port = train_data['embarked'].dropna().mode()[0]
freq_port


# In[183]:


for dataset in combine:
    dataset['embarked'] = dataset['embarked'].fillna(freq_port)
    
train_data[['embarked', 'survived']].groupby(['embarked'], as_index=False).mean().sort_values(by='survived', ascending=False)


# # Converting categorical feature to numeric

# In[184]:


for dataset in combine:
    dataset['embarked'] = dataset['embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_data.head()


# In[185]:


test_data['fare'].fillna(test_data['fare'].dropna().median(), inplace=True)
test_data.head()


# In[186]:


train_data['FareBand'] = pd.qcut(train_data['fare'], 4)
train_data[['FareBand', 'survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[187]:


for dataset in combine:
    dataset.loc[ dataset['fare'] <= 7.91, 'fare'] = 0
    dataset.loc[(dataset['fare'] > 7.91) & (dataset['fare'] <= 14.454), 'fare'] = 1
    dataset.loc[(dataset['fare'] > 14.454) & (dataset['fare'] <= 31), 'fare']   = 2
    dataset.loc[ dataset['fare'] > 31, 'fare'] = 3
    dataset['fare'] = dataset['fare'].astype(int)

train_data = train_data.drop(['FareBand'], axis=1)
combine = [train_data, test_data]
    
train_data.head(10)


# In[172]:


test_data.head(10)


# In[72]:


copy_df=train_data.copy()
copyTest_df=test_data.copy()


# # Encoding to numeric 

# In[73]:


from sklearn.preprocessing import OneHotEncoder


# In[74]:


train_Embarked = copy_df["embarked"].values.reshape(-1,1)
test_Embarked = copyTest_df["embarked"].values.reshape(-1,1)


# In[75]:


onehot_encoder = OneHotEncoder(sparse=False)
train_OneHotEncoded = onehot_encoder.fit_transform(train_Embarked)
test_OneHotEncoded = onehot_encoder.fit_transform(test_Embarked)


# In[76]:


copy_df["EmbarkedS"] = train_OneHotEncoded[:,0]
copy_df["EmbarkedC"] = train_OneHotEncoded[:,1]
copy_df["EmbarkedQ"] = train_OneHotEncoded[:,2]
copyTest_df["EmbarkedS"] = test_OneHotEncoded[:,0]
copyTest_df["EmbarkedC"] = test_OneHotEncoded[:,1]
copyTest_df["EmbarkedQ"] = test_OneHotEncoded[:,2]


# In[77]:


copy_df.head()


# In[78]:


copyTest_df.head()


# In[79]:


train_data.head()


# In[80]:


test_data.head()


# # Creating and Training a model

# In[81]:


X_trainTest = copy_df.drop(copy_df.columns[[0,5]],axis=1)
Y_trainTest = copy_df["survived"]
X_testTest = copyTest_df.drop(copyTest_df.columns[[0,5]],axis=1)
X_trainTest.head()


# In[82]:


X_testTest.head()


# In[88]:


logReg = LogisticRegression()
logReg.fit(X_trainTest,Y_trainTest)
acc = logReg.score(X_trainTest,Y_trainTest)
acc


# In[89]:


X_train = train_data.drop("survived", axis=1)
Y_train = train_data["survived"]
X_test  = test_data.drop("passengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
X_train.head()


# In[90]:


X_test.head()


# In[93]:


coeff_df = pd.DataFrame(train_data.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logReg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[95]:


logReg = LogisticRegression()
logReg.fit(X_train, Y_train)
acc_log = round(logReg.score(X_train, Y_train) * 100, 2)
acc_log


# In[96]:


svcTest = SVC()
svcTest.fit(X_trainTest, Y_trainTest)
acc_svcTest = round(svcTest.score(X_trainTest, Y_trainTest)*100,2)
acc_svcTest


# In[100]:


print("Support Vector Machines")

svc = SVC()
svc.fit(X_train, Y_train)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[98]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[101]:


print("Gaussian Naive Bayes")

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[104]:


print ("Stochastic Gradient Descent")

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[105]:


print("Decision Tree")

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[106]:


random_forestTest = RandomForestClassifier(n_estimators=100)
random_forestTest.fit(X_trainTest, Y_trainTest)
acc_random_forestTest = round(random_forestTest.score(X_trainTest, Y_trainTest) * 100, 2)
acc_random_forestTest


# In[107]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[108]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent',  
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian,
              acc_sgd,  acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




