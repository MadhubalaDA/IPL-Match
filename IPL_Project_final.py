#!/usr/bin/env python
# coding: utf-8

# In[138]:

import streamlit as st
import pandas as pd
import numpy as np

mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

delivery=pd.read_csv(r"E:\\Data Science\\Project\Machine Learning\\IPL\\SE\\deliveries.csv",encoding= 'unicode_escape')
delivery.head()


# In[140]:


match=pd.read_csv("E:\Data Science\Project\Machine Learning\IPL\SE\matches.csv")
match.head()


# In[141]:


delivery.shape


# In[142]:


delivery.describe


# In[143]:


delivery.info()


# In[144]:


match.shape


# In[145]:


match.describe


# In[146]:


match.info()


# In[147]:


total_run=delivery.groupby(['match_id','inning'])[['total_runs']].sum().reset_index()
total_run


# In[148]:


total_run=total_run[total_run['inning']==1]
total_run


# In[149]:


match_df=match.merge(total_run[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[150]:


match_df.head()


# In[151]:


match_df.team1.unique()


# In[152]:


match_df.replace(to_replace ="Delhi Daredevils",
                 value ="Delhi Capitals")


# In[153]:


match_df.replace(to_replace ="Deccan Chargers",value ="Sunrisers Hyderabad").head()


# In[154]:


len(match_df['toss_winner'])


# In[155]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[156]:


match_df.describe()


# In[157]:


toss_won=match_df['toss_winner'].value_counts().reset_index()
chart=sns.barplot(x="index", y="toss_winner", data=toss_won)
chart.tick_params(axis='x', rotation=90)
sns.set(rc = {'figure.figsize':(15,10)})
chart.set(xlabel='toss_winner',
       ylabel='Count',
       title='Maximum toss won')


# In[158]:


Total_win=match_df['winner'].value_counts().reset_index()
chart1=sns.barplot(x="index", y="winner", data=Total_win)
chart1.tick_params(axis='x', rotation=90)
sns.set(rc = {'figure.figsize':(15,10)})
chart1.set(xlabel='Total win by Teams',
       ylabel='Count',
       title='Maximum matches won')


# In[159]:


Toss_match_winner=match_df[match_df['toss_winner']==match_df['winner']]
slices=[len(Toss_match_winner),(756-len(Toss_match_winner))]
labels=['Yes','No']
plt.pie(slices,labels=labels,startangle=90,shadow=True,explode=(0,0),autopct='%1.2f%%',colors=['b','y'])
plt.title("Teams who had won Toss and Won the match")
fig = plt.gcf()
fig.set_size_inches(8,8)
plt.show()


# In[160]:


win_chart = match_df['winner'].value_counts()
print ('Team which won maximum number of matches in IPL :', win_chart.head(1))


# In[161]:


Toss_chart = match_df['toss_winner'].value_counts()
print ('Team which won maximum number of toss in IPL :', Toss_chart.head(1))


# In[162]:


delivery_df=match_df.merge(delivery,on='match_id')
delivery_df.head()


# In[163]:


delivery_df.info()


# In[164]:


delivery_df['match_id'].unique()


# In[165]:


delivery_df=delivery_df[delivery_df['inning']==2]
delivery_df


# In[166]:


delivery_df['current_score']=delivery_df.groupby(['match_id'])['total_runs_y'].cumsum()


# In[167]:


delivery_df.head()


# In[168]:


delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['current_score']
delivery_df.head()


# In[169]:


delivery_df['total_runs_x'] = delivery_df['total_runs_x'].astype(int)
delivery_df['current_score'] = delivery_df['current_score'].astype(int)


# In[170]:


delivery_df.info()


# In[171]:


delivery_df['Balls_left']=126-((delivery_df['over']*6)+delivery_df['ball'])
delivery_df.head()


# In[172]:


delivery_df[delivery_df['player_dismissed'].isnull()]


# In[173]:


delivery_df.player_dismissed.unique()


# In[174]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna(0)
delivery_df.player_dismissed.unique()


# In[177]:


delivery_df['current_runrate']=(delivery_df['current_score']*6)/(120-delivery_df['Balls_left'])
delivery_df.head()


# In[178]:


delivery_df['Required_runrate']=(delivery_df['runs_left']*6)/(delivery_df['Balls_left'])
delivery_df.head()


# In[179]:


delivery_df['Result_new'] = np.where(delivery_df['batting_team'] == delivery_df['winner'], 1,0)
delivery_df.Result_new.unique()


# In[180]:


delivery_df['wickets'] = np.where(delivery_df['player_dismissed'] == delivery_df['batsman'], 1,0)
delivery_df.wickets.unique()


# In[181]:


delivery_df.info()


# In[182]:


final_df=delivery_df.filter(['batting_team','bowling_team','city','runs_left','Balls_left','wickets','total_runs_x','current_runrate','Required_runrate','Result_new'])
final_df.head()


# In[183]:


final_df.shape


# In[184]:


final_df.isnull().sum()


# In[185]:


final_df=final_df.dropna()
final_df.info()


# In[186]:


final_df.shape


# In[187]:


final_df=final_df[final_df['Balls_left']!=0]
final_df


# In[220]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,train_size=0.85,random_state=1)


# In[189]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer


# In[190]:


numeric_features = ["runs_left", "Balls_left","wickets","total_runs_x","current_runrate","Required_runrate","Result_new"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["batting_team", "bowling_team", "city"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# In[191]:


final_df.head()


# In[192]:


new_df=final_df[['batting_team','bowling_team','city']]
new_df.head()


# In[193]:


encoder = OneHotEncoder(handle_unknown='ignore')
encoder_df = pd.DataFrame(encoder.fit_transform(new_df).toarray())
encoder_df.head()


# In[194]:


Final = final_df.join(encoder_df)
Final.head()


# In[215]:


Final.head()
Final.info()


# In[229]:


Y=Final[['Result_new']]
X=Final.drop(columns=['Result_new'])
X = np.nan_to_num(X)


# In[230]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,train_size=0.80,random_state=1)


# In[231]:


from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
pipe.fit(X_train, Y_train)
Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])
pipe.score(X_test, Y_test)


# In[232]:


Final.isnull().sum()


# In[233]:


Final = Final.fillna(0)
Final.isnull().sum()


# In[234]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=1000,random_state=42)
rf=model.fit(X_train,Y_train)
print('Yes')


# In[236]:


Y_test.head()


# In[260]:


Y_test['Prediction']=model.predict(X_test)
Y_test.head()


# In[264]:


from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(Y_test['Result_new'],Y_test['Prediction']))
print(accuracy_score(Y_test['Result_new'],Y_test['Prediction']))


# In[261]:


X_train1, X_test1, Y_train1, Y_test1=train_test_split(X,Y,train_size=0.85,random_state=1)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()


# In[265]:


model_LR=LR.fit(X_train1,Y_train1)
model_LR


# In[266]:


Y_test1.head()


# In[267]:


Y_test1['Predicted_result'] = model_LR.predict(X_test1)
Y_test1.head()


# In[268]:


pd.crosstab(index = Y_test1['Result_new'],columns = Y_test1['Predicted_result'], margins=True)


# In[269]:


print(confusion_matrix(Y_test1['Result_new'],Y_test1['Predicted_result']))
print(accuracy_score(Y_test1['Result_new'],Y_test1['Predicted_result']))


# In[270]:


import pickle


# In[274]:


filename='Predicted_result.pickle'
pickle.dump(model,open(filename,'wb'))


# In[ ]:




