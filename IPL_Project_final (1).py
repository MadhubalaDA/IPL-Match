#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


delivery=pd.read_csv("E:\Data Science\Project\Machine Learning\IPL\SE\deliveries.csv")
delivery.head()


# In[3]:


match=pd.read_csv("E:\Data Science\Project\Machine Learning\IPL\SE\matches.csv")
match.head()


# In[4]:


delivery.shape


# In[5]:


delivery.describe


# In[6]:


delivery.info()


# In[7]:


match.shape


# In[8]:


match.describe


# In[9]:


match.info()


# In[10]:


total_run=delivery.groupby(['match_id','inning'])[['total_runs']].sum().reset_index()
total_run


# In[11]:


total_run=total_run[total_run['inning']==1]
total_run['total_runs']=total_run['total_runs'].apply(lambda x:x+1)
total_run


# In[13]:


match_df=match.merge(total_run[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[14]:


match_df.head()


# In[15]:


match_df.team1.unique()


# In[27]:


match_df['team1']=match_df['team1'].str.replace("Delhi Daredevils","Delhi Capitals")
match_df['team2']=match_df['team2'].str.replace("Delhi Daredevils","Delhi Capitals")


# In[28]:


match_df['team1']=match_df['team1'].str.replace("Deccan Chargers","Sunrisers Hyderabad")
match_df['team2']=match_df['team2'].str.replace("Deccan Chargers","Sunrisers Hyderabad")


# In[29]:


match_df.team1.unique()


# In[30]:


teams= [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings', 
    'Rajasthan Royals',
    'Delhi Capitals'
]


# In[34]:


match_df.head()


# In[33]:


match_df.shape


# In[31]:


match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]
match_df['team1'].unique()


# In[37]:


match_df[match_df['dl_applied']==1].style.background_gradient(cmap='plasma')


# In[38]:


match_df=match_df[match_df['dl_applied']==0]
match_df=match_df[['match_id','city','winner','total_runs']]
match_df


# In[39]:


delivery.head()


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[41]:


match_df.describe()


# In[43]:


Total_win=match_df['winner'].value_counts().reset_index()
chart1=sns.barplot(x="index", y="winner", data=Total_win)
chart1.tick_params(axis='x', rotation=90)
sns.set(rc = {'figure.figsize':(15,10)})
chart1.set(xlabel='Total win by Teams',
       ylabel='Count',
       title='Maximum matches won')


# In[45]:


win_chart = match_df['winner'].value_counts()
print ('Team which won maximum number of matches in IPL :', win_chart.head(1))


# In[50]:


delivery_df=match_df.merge(delivery,on='match_id')
delivery_df.head()


# In[51]:


delivery_df.info()


# In[52]:


delivery_df['match_id'].unique()


# In[53]:


delivery_df=delivery_df[delivery_df['inning']==2]
delivery_df


# In[54]:


delivery_df['current_score']=delivery_df.groupby('match_id').cumsum()['total_runs_y']


# In[55]:


delivery_df.head()


# In[56]:


delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['current_score']
delivery_df.head()


# In[32]:


delivery_df['total_runs_x'] = delivery_df['total_runs_x'].astype(int)
delivery_df['current_score'] = delivery_df['current_score'].astype(int)


# In[33]:


delivery_df.info()


# In[57]:


delivery_df['Balls_left']=126-((delivery_df['over']*6)+delivery_df['ball'])
delivery_df.head()


# In[58]:


list(delivery_df['player_dismissed'].unique())[:2]


# In[59]:


delivery_df['player_dismissed']=delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed']=delivery_df['player_dismissed'].apply(lambda x:x if x=="0" else "1")
delivery_df['player_dismissed']=delivery_df['player_dismissed'].astype('int')


# In[60]:


delivery_df['player_dismissed'].unique()


# In[61]:


wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets']=10-wickets
delivery_df


# In[62]:


delivery_df['current_runrate']=(delivery_df['current_score']*6)/(120-delivery_df['Balls_left'])
delivery_df.head()


# In[63]:


delivery_df['Required_runrate']=(delivery_df['runs_left']*6)/(delivery_df['Balls_left'])
delivery_df.head()


# In[64]:


def resultfun(row):
    return 1 if row['batting_team'] == row['winner'] else 0


# In[65]:


delivery_df['Result_new'] = delivery_df.apply(resultfun,axis=1)
delivery_df.head()


# In[71]:


sns.countplot(delivery_df['Result_new'])


# In[73]:


final_df=delivery_df[['batting_team','bowling_team','city','runs_left','Balls_left','wickets','total_runs_x','current_runrate','Required_runrate','Result_new']]
final_df.head()


# In[74]:


final_df.shape


# In[75]:


final_df.isnull().sum()


# In[76]:


final_df=final_df.dropna()
final_df.isnull().sum()


# In[78]:


final_df.shape


# In[77]:


final_df=final_df[final_df['Balls_left']!=0]
final_df


# In[79]:


data = final_df.copy()
test=data['Result_new']
train = data.drop(['Result_new'],axis=1)
train.head()


# In[80]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(train,test,train_size=0.80,random_state=2408)


# In[88]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# In[83]:


cf= ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),["batting_team", "bowling_team", "city"])
],remainder='passthrough')


# In[87]:


pipe = Pipeline(steps=[
    ('step1',cf),
    ('step2',LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train, Y_train)


# In[89]:


y_pred=pipe.predict(X_test)
print(metrics.accuracy_score(Y_test,y_pred))


# In[90]:


pipe.predict_proba(X_test)[10]


# In[91]:


from sklearn.ensemble import RandomForestClassifier
pipe2 = Pipeline(steps=[
    ('step1',cf),
    ('step2',RandomForestClassifier( ))
])

pipe2.fit(X_train, Y_train)
print(metrics.accuracy_score(Y_test,pipe2.predict(X_test)))


# In[92]:


pipe2.predict_proba(X_test)[10]


# In[93]:


import pickle


# In[96]:


pickle.dump(pipe,open('Pipe.pkl','wb'))


# In[ ]:




