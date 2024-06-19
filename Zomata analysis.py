#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataframe = pd.read_csv("zomato (1).csv")
print(dataframe.head())


# In[3]:


def handleRate(value):
    value=str(value).split('/')
    value=value[0];
    return float(value)
 
dataframe['Aggregate rating']=dataframe['Aggregate rating'].apply(handleRate)
print(dataframe.head())


# In[4]:


dataframe.info()


# In[5]:


sns.countplot(x=dataframe['Cuisines'])
plt.xlabel("Type of restaurant")


# In[6]:


grouped_data = dataframe.groupby('Cuisines')['Votes'].sum()
result = pd.DataFrame({'votes': grouped_data})
plt.plot(result, c="green", marker="o")
plt.xlabel("Type of restaurant", c="red", size=20)
plt.ylabel("Votes", c="red", size=20)


# In[7]:


# the restaurant’s name that received the maximum votes based on a given dataframe

max_votes = dataframe['Votes'].max()
restaurant_with_max_votes = dataframe.loc[dataframe['Votes'] == max_votes, 'Restaurant Name']
 
print("Restaurant(s) with the maximum votes:")
print(restaurant_with_max_votes)


# In[8]:


# Online_order column

sns.countplot(x=dataframe['Online'])


# In[10]:


# Rate column

plt.hist(dataframe['Aggregate rating'],bins=5)
plt.title("Ratings Distribution")
plt.show()


# In[11]:


#Let’s see the approx_cost(for two people) column.

couple_data=dataframe['Average Cost for two']
sns.countplot(x=couple_data)


# In[13]:


#Now we will see whether online orders receive higher ratings than offline orders.

plt.figure(figsize = (6,6))
sns.boxplot(x = 'Online', y = 'Aggregate rating', data = dataframe)


# In[16]:


pivot_table = dataframe.pivot_table(index='Cuisines', columns='Online', aggfunc='size', fill_value=0)
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt='d')
plt.title("Heatmap")
plt.xlabel("Online")
plt.ylabel("Cuisines")
plt.show()


# In[21]:


#Conclusion: 

 #Dining establishments primarily accept offline orders, while cafés primarily take internet orders. This implies that customers prefer to place orders in person at restaurants but order online at cafés.


# In[ ]:




