#!/usr/bin/env python
# coding: utf-8

# ## IMPORT ALL REQUIRED LYBRIRES HERE WE ARE DRAWING THE FIGURES USING PLOTLY LIBRARY

# In[99]:


import pandas as pd


# In[100]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().system('pip install plotly')
import plotly.express as ex


# In[101]:


import plotly.graph_objects as go

from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
import warnings
warnings.filterwarnings('ignore')


# ## IMPORT DATA HERE WE ARE IMPORTED DATA FROM DESKTOP

# In[102]:


dt = pd.read_csv(r'C:\Users\chand\Downloads\top-500-movies.csv')


# In[103]:


dt.head()


# ## HERE WE ARE DOING THE EXPLORATORY  DATA  ANALYSIS

# In[104]:


dt.shape


# In[105]:


dt.sample(5)


# In[106]:


dt.isnull().sum()


# In[107]:


dt.dropna(inplace=True
         )


# In[108]:


dt.isnull().sum()


# In[109]:


dt.head()


# In[110]:


dt.shape


# In[111]:


dt.info()


# ## HERE THE DATE TIME LINE IS IN OBJECTIVE FORM SO WE ARE CONVERTING THE FORMAT INTO THE DATE FORMAT

# In[112]:


dt['release_date'] = pd.to_datetime(dt['release_date'])


# In[113]:


dt.duplicated().sum()


# ## here we are taking the movie released year 

# In[114]:


dt['year'] = dt['release_date'].dt.year


# In[115]:


dt['year']


# In[116]:


dt['month'] = dt['release_date'].dt.month


# In[117]:


dt['month']


# ## HERE WE HAVE TO CHECK THE PROFIT OF MOVIES

# In[118]:


dt['profit'] = dt['worldwide_gross'] - dt['production_cost']


# In[119]:


def best_revenue(dt, n=10, year =0):
    if year == 0:
        dt= dt.sort_values(by = 'profit', ascending = False).head(n)
        title = 'Best of all time'
    else:
        title = f'Best of {year}'
        dt= dt[dt['year']==year].sort_values(by='profit', ascending = False).head(n)
    fig = ex.bar(dt, x='profit', y ='title',orientation='h',title = title, text = 'year')
    fig.update_layout(uniformtext_minsize=8,uniformtext_mode = 'hide')
    fig.show()


# In[120]:


best_revenue(dt)


# In[121]:


a = dt.sort_values(by='profit', ascending=False)


# In[122]:


a


# In[123]:


v = dt[dt['year']== 2022].sort_values(by='profit', ascending = False)


# In[124]:


v


# In[125]:


best_revenue(dt, year=2022)


# In[126]:


best_revenue(dt, year=2021)


# In[127]:


def best_genre(dt,n=10,year=0):
    if year == 0:
        dt = dt.groupby('genre').sum().sort_values(by='profit',ascending = False).head(n)
        title = 'Best Genre of all time'
    else:
        dt = dt[dt['year'] == year].groupby('genre').sum().sort_values(by='profit', ascending = False).head(n)
        title = f'Best genre of {year}'
    fig = ex.bar(dt, x = 'profit', y=dt.index, orientation='h',title = title)
    fig.update_layout(uniformtext_minsize=8,uniformtext_mode = 'hide')
    fig.show()
    
best_genre(dt)


# In[128]:


best_genre(dt, year=2021)


# In[129]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


# In[130]:


dt.columns


# In[131]:


dt.drop(['rank', 'release_date', 'title', 'url', 'domestic_gross','opening_weekend', 'year', 'month', 'profit'], inplace =True, axis=1)


# In[132]:


dt.head()


# In[133]:


categorical_cols =dt.select_dtypes(include='object').columns


# In[134]:


categorical_cols


# In[135]:


le = LabelEncoder()


# In[136]:


for col in categorical_cols:
    dt[col]= le.fit_transform(dt[col])


# In[137]:


dt


# In[138]:


x = dt.drop('worldwide_gross', axis =1)
y =dt['worldwide_gross']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2, random_state=42)

model = RandomForestRegressor(n_estimators =100, random_state = 42 )
model.fit(x_train,y_train)


# In[139]:


y_pred = model.predict(x_test)


# In[140]:


import numpy as np


# In[141]:


print('MSE :', mean_squared_error(y_pred,y_test).round())
print('MAS :', mean_absolute_error(y_pred,y_test).round(2))
print('RMSE:', np.sqrt(mean_squared_error(y_pred,y_test)).round(2))


# In[142]:


model.score(x_test, y_test)


# In[143]:


feature_importance = pd.DataFrame({'feature': x.columns, 'importance': model.feature_importances_})


# In[144]:


feature_importance 


# In[145]:


feature_importance.sort_values(by='importance', ascending=False)


# In[146]:


fig = ex.bar(feature_importance.sort_values(by='importance'), x='importance', y='feature', orientation='h')
fig.show()


# In[ ]:




