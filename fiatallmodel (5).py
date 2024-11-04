#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#pip install pandas


# In[3]:


data=pd.read_csv(r"fiat500.csv")


# In[4]:


print("manu chetta")


# In[5]:


data.shape


# In[6]:


data.tail(10)


# In[7]:


data.describe()


# In[8]:


list(data)


# In[9]:


data.tail(10)


# In[10]:


data["model"].unique()


# In[ ]:





# In[11]:


data.info()


# In[12]:


data.groupby(['model']).count()


# In[13]:


data.groupby(['previous_owners']).count()


# In[14]:


data['model'].unique()


# In[15]:


data.shape
#df=data
#data=df.loc[(df.model=='lounge')&(df.previous_owners==1)]


# In[16]:


cor=data.corr()
cor
#cor is alays bw -1 and 1


# In[17]:


#correlation metrix 
import seaborn as sns
sns.heatmap(cor,vmax=1,vmin=-1,annot=True,linewidths=.5,cmap='bwr')


# In[21]:


data1=data.drop(['lat','ID'],axis=1) #unwanted columns removed


# In[22]:


#2-3
data2=data1.drop('lon',axis=1)


# In[23]:


data2.shape


# In[24]:


data2.head(3)


# In[25]:


data2=pd.get_dummies(data2,dtype=int)


# In[26]:


#data2.groupby(['previous_owners'])
data2.shape


# In[27]:


data2.head(5)


# In[28]:


data.head(5)


# In[29]:


y=data2['price']
X=data2.drop('price',axis=1)


# In[30]:


y


# In[31]:


X 


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=0) #0.67 data will be for training.
#X_train having 67 %data with out price
#X_test         33 
#y_train        67 % dat only with price
#y_test         33 % data only with price


# In[33]:


X_train.head(50)


# In[34]:


X_test.head(5)


# In[35]:


X_train.shape


# In[36]:


y_train.shape


# In[37]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression() #creating object of LinearRegression
reg.fit(X_train,y_train) #training and fitting LR object using training data


# In[38]:




#X_test=[[51,2197,70000,1,1,0,0],[51,3127,100000,1,1,0,0],[51,5227,175000,1,1,0,0]]


# In[39]:


#above line to actual


# In[40]:


ypred=reg.predict(X_test) 


# In[41]:


ypred


# In[42]:


# filename='pricemodeldummy1'
# pickle.dump(reg,open(filename,'wb'))


# In[ ]:





# In[43]:


#savedmodel=pickle.load(open(filename,'rb'))

#X_test=[[1,75,1062,8000,1]]
#savedmodel.predict(X_test)


# In[44]:


from sklearn.metrics import r2_score
r2_score(y_test,ypred)


# In[45]:


from sklearn.metrics import mean_absolute_percentage_error as mape

mape_value = mape(y_test, ypred)
mape_value


# In[46]:


from sklearn.metrics import mean_squared_error #calculating MSE
mean_squared_error(y_test,ypred)
#print(t**.5)


# In[ ]:





# In[47]:


#Results= pd.DataFrame(columns=['Actual','Predicted'])
#Results['Actual']=y_test
Results= pd.DataFrame(columns=['Price','Predicted'])
Results['Price']=y_test
Results['Predicted']=ypred
#Results['km']=X_test['km']
Results=Results.reset_index()
Results['Id']=Results.index
Results.head(15)


# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(x='Id',y='Price',data=Results.head(50)) #actual
sns.lineplot(x='Id',y='Predicted',data=Results.head(50)) #predicted
plt.plot()


# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(x='Id',y='Price',data=Results.tail(50))
sns.lineplot(x='Id',y='Predicted',data=Results.tail(50))
plt.plot()


# In[50]:


list(X_test)


# In[51]:


# Saving model to disk
pickle.dump(reg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# # this is for prediction of a new veicle with spec

# In[52]:


new=[[51,3000,65000,1,1,0,0]]


# In[53]:


real=reg.predict(new)


# In[54]:


real

#------------------------------------------------------------------#
# In[55]:


# ridge regression


# In[60]:


from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV


from sklearn.linear_model import Ridge

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20,30]

ridge = Ridge()

parameters = {'alpha': alpha}

ridge_regressor = GridSearchCV(ridge, parameters)

ridge_regressor.fit(X_train, y_train)


# In[61]:


# ridge_regressor.best_params_


# In[ ]:


#X_train=[2]


# In[ ]:


# ridge=Ridge(alpha=30)
# ridge.fit(X_train,y_train)
# y_pred_ridge=ridge.predict(X_test)


# In[ ]:


# from sklearn.metrics import mean_squared_error
# Ridge_Error=mean_squared_error(y_pred_ridge,y_test)
# Ridge_Error


# In[ ]:


# from sklearn.metrics import r2_score
# r2_score(y_test,y_pred_ridge)


# In[ ]:


# from sklearn.metrics import mean_absolute_percentage_error as mape

# mape_value = mape(y_test,y_pred_ridge)
# mape_value


# In[ ]:


# Results= a.DataFrame(columns=['Actual','Predicted_ridge'])
# Results['Actual']=y_test
# Results['Predicted_ridge']=y_pred_ridge
# #Results['km']=X_test['km']
# Results=Results.reset_index()
# Results['Id']=Results.index
# Results.head(10)


# In[ ]:


# sns.lineplot(x='Id',y='Actual',data=Results.head(50))
# sns.lineplot(x='Id',y='Predicted_ridge',data=Results.head(50))
# plt.plot()


# In[ ]:


# sns.lineplot(x='Id',y='Actual',data=Results.tail(50))
# sns.lineplot(x='Id',y='Predicted_ridge',data=Results.tail(50))
# plt.plot()


# In[ ]:


#elastic


# In[ ]:


# from sklearn.linear_model import ElasticNet

# elastic = ElasticNet()

# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

# elastic_regressor = GridSearchCV(elastic, parameters)

# elastic_regressor.fit(X_train, y_train)


# In[ ]:


elastic_regressor.best_params_


# In[ ]:


# elastic=ElasticNet(alpha=1)
# elastic.fit(X_train,y_train)
# y_pred_elastic=elastic.predict(X_test)


# In[ ]:


# from sklearn.metrics import r2_score
# r2_score(y_test,y_pred_elastic)


# In[ ]:


# mape_value = mape(y_test,y_pred_elastic)
# mape_value


# In[ ]:


# elastic_Error=mean_squared_error(y_pred_elastic,y_test)
# elastic_Error


# In[ ]:


# Results= a.DataFrame(columns=['Actual','Predicted'])
# Results['Actual']=y_test
# Results['Predicted']=y_pred_elastic
# #Results['km']=X_test['km']
# Results=Results.reset_index()
# Results['Id']=Results.index
# Results.head(10)


# In[ ]:


# sns.lineplot(x='Id',y='Actual',data=Results.head(50))
# sns.lineplot(x='Id',y='Predicted',data=Results.head(50))
# plt.plot()


# In[ ]:


#RandomForest
# X_train.shape


# In[ ]:


# from sklearn.model_selection import GridSearchCV #GridSearchCV is for parameter tuning
# from sklearn.ensemble import RandomForestRegressor
# reg=RandomForestRegressor()
# n_estimators=[25,50,75,100,125,150,175,200] #number of decision trees in the forest, default = 100
# criterion=['mse'] #criteria for choosing nodes default = 'gini'
# max_depth=[3,5,10] #maximum number of nodes in a tree default = None (it will go till all possible nodes)
# parameters={'n_estimators': n_estimators,'criterion':criterion,'max_depth':max_depth}  
# RFC_reg = GridSearchCV(reg, parameters)
# RFC_reg.fit(X_train,y_train)


# In[ ]:


# RFC_reg.best_params_


# In[ ]:


# reg=RandomForestRegressor(n_estimators=50,criterion='mse',max_depth=5)


# In[ ]:


# reg.fit(X_train,y_train)


# In[ ]:


# y_pred=reg.predict(X_test)


# In[ ]:


# y_pred


# In[ ]:


# from sklearn.metrics import r2_score
# r2_score(y_test,y_pred)


# In[ ]:


# mape_value = mape(y_test,y_pred)
# mape_value


# In[ ]:


# Results= a.DataFrame(columns=['Actual','Predicted'])
# Results['Actual']=y_test
# Results['Predicted']=y_pred
# #Results['km']=X_test['km']
# Results=Results.reset_index()
# Results['Id']=Results.index
# Results.head(10)


# In[ ]:


# sns.lineplot(x='Id',y='Actual',data=Results.head(100))
# sns.lineplot(x='Id',y='Predicted',data=Results.head(100))
# plt.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




