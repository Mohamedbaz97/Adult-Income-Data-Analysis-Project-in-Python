#!/usr/bin/env python
# coding: utf-8

# ## Adult Income Project

# ### For this project, I'll be using the "Adult Income" dataset from the UCI Machine Learning Repository. This dataset contains information about individuals' demographics, education, and income, and our goal will be to use this data to predict whether an individual makes over $50,000 per year.

# #### 1- Load the dataset into a dataframe

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load the data into a pandas DataFrame
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, names=columns)
c=pd.read_csv(url)
print(c)


# #### 2- Drop any rows with missing values

# In[5]:


data = data.replace('?', np.nan)
data = data.dropna()


# #### 3- Convert the target variable to a binary classification problem

# In[6]:


data['income'] = (data['income'] == ' >50K')


# #### 4- Convert categorical variables to numerical

# In[7]:


cat_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for column in cat_columns:
    data[column] = pd.Categorical(data[column])
    data[column] = data[column].cat.codes


# #### 5- Split the data into training and testing sets

# In[8]:


X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### 6- Train a logistic regression model

# In[9]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# #### 7- Evaluate the model on the testing set

# In[10]:


y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# #### 8- Plot a histogram of the predicted probabilities for the positive class

# In[12]:


probs = model.predict_proba(X_test)[:,1]
sns.histplot(probs[y_test == 1], kde=False, label='Positive class', color='green')
sns.histplot(probs[y_test == 0], kde=False, label='Negative class', color='red')
plt.legend()
plt.xlabel('Predicted probability of positive class')
plt.ylabel('Count')
plt.show()


# #### 9- Visualize the number of individuals who make over $50,000 per year and those who do not.

# In[13]:


sns.countplot(data=data, x='income')
plt.xlabel('Income')
plt.ylabel('Count')
plt.show()


# #### 10- Visualize the distribution of age for individuals who make over $50,000 per year and those who do not.

# In[14]:


sns.boxplot(data=data, x='income', y='age')
plt.xlabel('Income')
plt.ylabel('Age')
plt.show()


# #### 11- Visualize the number of individuals who make over $50,000 per year by education level.

# In[15]:


sns.countplot(data=data, x='education', hue='income')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.legend(title='Income', loc='upper right')
plt.xticks(rotation=90)
plt.show()


# #### 12- Visualize the trade-off between true positive rate and false positive rate for different thresholds of the predicted probabilities.

# In[16]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# #### We can see that our model is performing better than random, with an AUC of 0.73. However, there is still room for improvement, as the curve could be closer to the top-left corner of the plot.

# #### 13- Visualize the number of individuals who make over $50,000 per year by occupation.

# In[17]:


sns.countplot(data=data, x='occupation', hue='income')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.legend(title='Income', loc='upper right')
plt.xticks(rotation=90)
plt.show()


# #### 14- Visualize the number of males and females who make over $50,000 per year.

# In[18]:


sns.countplot(data=data, x='sex', hue='income')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Income', loc='upper right')
plt.show()


# ### Based on our analysis we can see that:
# #### 1- Income is positively associated with education level, work experience, and work hours per week. Individuals with higher education levels, more work experience, and longer work hours tend to make more money.
# 
# #### 2- Males are more likely to make over $50,000 per year compared to females, and individuals who are married with a spouse present are more likely to make over $50,000 per year compared to other marital status categories.
# 
# #### 3- There are some occupations, such as "Exec-managerial", "Prof-specialty" and "Sales", that have a higher proportion of individuals making over $50,000 per year compared to other occupations.
