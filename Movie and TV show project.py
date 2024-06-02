#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('netflix_titles.csv')


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.tail()     #to show bottom 5 records of dataset


# In[6]:


data.columns


# In[7]:


data.info()


# # we check the duplicate values in our dataset 

# In[8]:


data[data.duplicated()]      #to check the douplicate rows in the dataset


# #No douplicate value in the dataset
# #After this we check the null value in the dataset

# In[9]:


data.isnull()


# In[10]:


#to show the null value in the each columns


# In[11]:


data.isnull().sum()


# In[12]:


data = data.dropna()


# In[13]:


import seaborn as sns


# In[14]:


sns.heatmap(data.isnull())


# Q.1. In which year highest number of the TV Shows & Movies were released?

# In[15]:


#this question is related to the datetime formate then we check the type of dataset 


# In[16]:


data.dtypes


# In[17]:


data['rating'].nunique()


# #we convert the 'release_year' column into the datetime formate

# In[18]:


data['Date_N'] = pd.to_datetime(data['date_added'])


# In[19]:


data.dtypes


# #then we show the year wise result

# In[20]:


data.head()


# In[ ]:





# In[21]:


data['Date_N'].dt.year.value_counts()


# #then we show it in the plot

# In[44]:


data['Date_N'].dt.year.value_counts().plot(kind = 'bar')


# Q.2. How many Movies & TV Shows are in the dataset?

# 

# In[23]:


data.groupby('type').type.count()


# In[ ]:





# In[24]:


data.groupby('type').type.count().plot(kind = 'pie')


# Q.3. which country has the highest no. of TV Shows?

# In[ ]:


data_tvshow = data[data['type'] == 'TV Show']


# In[31]:


data_tvshow.country.value_counts()


# In[42]:


data_tvshow.country.value_counts().plot(kind = 'bar')


# Q.4. which country has the highest no. of Movies?

# In[29]:


data_Movie = data[data['type'] == 'Movie']


# In[30]:


data_Movie.country.value_counts()


# Q.5. show that the movies & tv show where realsed in year 2021?

# In[39]:


data[ (data['type'] == 'Movie') & (data['release_year'] == 2021) ]


# In[38]:


data[ (data['type'] == 'TV Show') & (data['release_year'] == 2021) ]


# # conclusion

# #we predict that this dataset first the highest movies and the tv show where realsed in which year.

# #second predict the number of movies and tv shows in the dataset.

# #third predict that the which country realsed the highest tv shows and the highest movies.

# #And also predict that the number of movies and tv shows where realsed in the 2021.

# #import the seabron library to draw the plots.
