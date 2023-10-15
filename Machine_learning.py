#Importig Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


#df is the cleaned dataset that was saved from the cleaning notebook
df=pd.read_csv('cleaned_dataset.csv')

#y_colummns presents our targets variables
y_columns=df.loc[:,['TotalGHGEmissions','SiteEnergyUse(kBtu)']]


#Let's take a look into our y_columns
print('Y:\n',y_columns.head())


#X presents our input variables
X=df.drop(['TotalGHGEmissions','SiteEnergyUse(kBtu)'],axis=1)
print('X:\n', X.head())
#to_categorical variable presents the categorical data that we should convert them to be provided to machine and deep learning algorithms which in turn improve predictions
to_categorical = X.select_dtypes(['object']).keys()
list(to_categorical)


#perform get_dummies in our categorical variables
dummy= pd.get_dummies(data=X, columns=list(to_categorical))



#we should drop our old categorical variables
X=pd.concat([X,dummy],axis=1)
X.drop(to_categorical,axis=1,inplace=True)


#Function that plot the comparision betweeen the true values and the predictions ones
def y_analysis_comparision(string,y_true1,y_pred1):
    
    sns.scatterplot(y_true1, y_pred1, alpha=0.2,color='r')
    #plot the line when y_pred=y_true
    plt.plot(y_true1, y_true1, '-k')
    plt.title(string, fontweight="bold")
    plt.xlabel("y_true", fontweight="bold")
    plt.ylabel("y_pred", fontweight="bold")
    





#Random Forest Model
#scores_rf presents our scores metrics dictionary for the random forest model
scores_dict_rf={
        'Train MS Error':[],
        'Train Score':[],
        'Test MS Error':[],
        'Test Score':[]}
#y_test_rf presents the test data of the random forest model
y_test_rf={}

#y_pred_rf presents the predictions of the random forest model
y_pred_rf={}


for column in y_columns.columns:
    
    #Splitting the data 
    X_train, X_test, y_train, y_test = train_test_split(X,y_columns[column],test_size=0.2,random_state=0)
    
    #scaler presents our scaler for standatization
    scaler=StandardScaler()
    
    # Normalize numerical features using  StandardScaler()
    numeric_columns1 = X_train.select_dtypes(['float64', 'int64']).columns
    X_train[numeric_columns1] = scaler.fit_transform(X_train[numeric_columns1],y_train)
    
    # use training set parameters to normalize test set. Model should not have any info about test set
    numeric_columns2 = X_test.select_dtypes(['float64', 'int64']).columns
    X_test[numeric_columns2] = scaler.transform(X_test[numeric_columns2])
    
    #for preventing errors
    X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    y_train = y_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    
    
    #Random Forest Model
    RandomForestmodel_search = RandomForestRegressor(random_state=0)
    
    #fit the model
    RandomForestmodel_search.fit(X_train, y_train)
      
    #Train Prediction
    pred_train_rf=RandomForestmodel_search.predict(X_train)
    
    #Test Prediction
    pred_test_rf = RandomForestmodel_search.predict(X_test)
    
    #Add te values to the dict
    y_test_rf[column]=y_test
    y_pred_rf[column]=pred_test_rf
    
    
    #train_rmse presents the mean squared error for  training
    train_rmse = np.sqrt(mean_squared_error(y_train,pred_train_rf))
    
    #test_rmse presents the mean squared error for  testing
    test_rmse = np.sqrt(mean_squared_error(y_test,pred_test_rf))
    
    #score_train presents the score of the training 
    score_train=r2_score(y_train,pred_train_rf)*100
    
    #score_test presents the score of the testing 
    score_predict=r2_score(y_test,pred_test_rf)*100

    #Append the scores metrics in the dictionary 
    scores_dict_rf["Train MS Error"].append(train_rmse)
    scores_dict_rf["Train Score"].append(score_train)
    scores_dict_rf["Test MS Error"].append(test_rmse)
    scores_dict_rf["Test Score"].append(score_predict)

    
#scores_rf presents the table of scores metrics of Random Forest Model
print('Scores of Random Forest Model :\n')
scores_rf=pd.DataFrame(scores_dict_rf)
scores_rf.index=["TotalGHGEmissions","SiteEnergyUse(kBtu)"]
print(scores_rf)

#Now we need to plot the scores of Random Forest for Total GHGEmissions and Energy Consumption
fig = plt.figure(figsize=(7, 6))
plt.title('Scores Of Random Forest Model',fontsize=18)
plt.bar(scores_rf.index,scores_rf['Test Score'],color=['#CD5C5C'])
plt.ylabel('Scores',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ticks = [20,30,40,50,60,70,80,90,100]
plt.yticks(ticks)
plt.show()


#Comparison of Actual vs Fitted Values For Random Forest Model
plt.figure(figsize=(15,7))
plt.suptitle('Comparison of Actual vs Fitted Values For Random Forest Model',fontweight="bold",size=17)

plt.subplot(1,2,1)
y_analysis_comparision('TotalGHG Emissions',list(y_test_rf.values())[0],list(y_pred_rf.values())[0])

plt.subplot(1,2,2)
y_analysis_comparision('SiteEnergyUse(kBtu)',list(y_test_rf.values())[1],list(y_pred_rf.values())[1])










#Gradient Boosting Model
#scores_gb presents our scores metrics dictionary for the gradient boosting model
scores_dict_gb={
        'Train MS Error':[],
        'Train Score':[],
        'Test MS Error':[],
        'Test Score':[]}

#y_test_gb presents our test data for the gradient boosting model
y_test_gb={}

#y_pred_gb presents our predictions for the gradient boosting model
y_pred_gb={}

for column in y_columns.columns:
    
    #Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X,y_columns[column],test_size=0.2,random_state=20)
    
    #scaler presents our scaler for standartization
    scaler=StandardScaler()
    
    # Normalize numerical features using sklearn scaler
    numeric_columns1 = X_train.select_dtypes(['float64', 'int64']).columns
    X_train[numeric_columns1] = scaler.fit_transform(X_train[numeric_columns1],y_train)
    
    # use training set parameters to normalize test set. Model should not have any info about test set
    numeric_columns2 = X_test.select_dtypes(['float64', 'int64']).columns
    X_test[numeric_columns2] = scaler.transform(X_test[numeric_columns2])
    
    #for preventing errors
    X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    y_train = y_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    y_test = y_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    
    #Gradient Boosting model
    Gbr=GradientBoostingRegressor(random_state=20)
  
    #fit the model
    Gbr.fit(X_train, y_train)
        
    #Train Prediction
    pred_train_gb= Gbr.predict(X_train)
    
    #Test Prediction
    pred_test_gb=Gbr.predict(X_test)
    
    #Add the values to the dict
    y_test_gb[column]=y_test
    y_pred_gb[column]=pred_test_gb
    
    #train_rmse presents the mean squared error for  training
    train_rmse = np.sqrt(mean_squared_error(y_train,pred_train_gb))
    
    #test_rmse presents the mean squared error for  testing
    test_rmse = np.sqrt(mean_squared_error(y_test,pred_test_gb))
    
    #score_train presents the score of the training 
    score_train=r2_score(y_train,pred_train_gb)*100
    
    #score_test presents the score of the testing 
    score_predict=r2_score(y_test,pred_test_gb)*100

    #Append the scores metrics in the dictionary
    scores_dict_gb["Train MS Error"].append(train_rmse)
    scores_dict_gb["Train Score"].append(score_train)
    scores_dict_gb["Test MS Error"].append(test_rmse)
    scores_dict_gb["Test Score"].append(score_predict)


#scores_GB presents the table of scores metrics of Gradient Boosting Model
print('Scores of Gradient Boosting Model :\n')
scores_GB=pd.DataFrame(scores_dict_gb)
scores_GB.index=["TotalGHGEmissions","SiteEnergyUse(kBtu)"]
print(scores_GB)


#Now we need to plot the scores of Gradient Boosting for Total GHGEmissions and Energy Consumption
fig = plt.figure(figsize=(7, 6))
plt.title('Scores Of Gradient Boosting Model',fontsize=18)
plt.bar(scores_GB.index,scores_GB['Test Score'],color='#CD5C5C')
plt.ylabel('Scores',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ticks = [20,30,40,50,60,70,80,90,100]
plt.yticks(ticks)
plt.show()


#Now we plot the Comparison of Actual vs Fitted Values For Gradient Boosting Model 
plt.figure(figsize=(15,7))
plt.suptitle('Comparison of Actual vs Fitted Values For Gradient Boosting Model',fontweight="bold",size=17)
plt.subplot(1,2,1)
y_analysis_comparision('TotalGHG Emissions',list(y_test_gb.values())[0],list(y_pred_gb.values())[0])
plt.subplot(1,2,2)
y_analysis_comparision('SiteEnergyUse(kBtu)',list(y_test_gb.values())[1],list(y_pred_gb.values())[1])









#Lasso Model
#scores_dict_lasso presents the scores metrics of the lasso model
scores_dict_lasso={
        'Train MS Error':[],
        'Train Score':[],
        'Test MS Error':[],
        'Test Score':[]}
#y_test_lasso presents the test data of the lasso model
y_test_lasso={}

#y_pred_lasso presents the predictions of th lasso model
y_pred_lasso={}

for column in y_columns.columns:
    #Splitting our  data 
    X_train, X_test, y_train, y_test = train_test_split(X,y_columns[column],test_size=0.2,random_state=0)
    
    #Scaler is our sclaer for standartization
    scaler=StandardScaler()
    
    # Normalize numerical features in train data using sklearn scaler
    numeric_columns1 = X_train.select_dtypes(['float64', 'int64']).columns
    X_train[numeric_columns1] = scaler.fit_transform(X_train[numeric_columns1],y_train)
    
    # # Normalize numerical features in test data using sklearn scaler
    numeric_columns2 = X_test.select_dtypes(['float64', 'int64']).columns
    X_test[numeric_columns2] = scaler.transform(X_test[numeric_columns2])
    
    #for preventing error
    X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    y_train = y_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    y_test = y_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    
    #Model Lasso
    Lasr=Lasso(random_state=0)
   
    #Fit The model
    Lasr.fit(X_train, y_train)
    
    
    #Train Prediction
    pred_train_lasso =Lasr.predict(X_train)
    
    #Test Prediction
    pred_test_lasso = Lasr.predict(X_test)
    
    #append in the dict
    y_test_lasso[column]=y_test
    y_pred_lasso[column]=pred_test_lasso
    
    #train_rmse presents the mean squared error for  training
    train_rmse = np.sqrt(mean_squared_error(y_train,pred_train_lasso))
    
    #test_rmse presents the mean squared error for  testing
    test_rmse = np.sqrt(mean_squared_error(y_test,pred_test_lasso))
    
    #score_train presents the score of the training 
    score_train=r2_score(y_train,pred_train_lasso)*100
    
    #score_test presents the score of the testing 
    score_predict=r2_score(y_test,pred_test_lasso)*100
    

    #Append in the scores metrics dictionary
    scores_dict_lasso["Train MS Error"].append(train_rmse)
    scores_dict_lasso["Train Score"].append(score_train)
    scores_dict_lasso["Test MS Error"].append(test_rmse)
    scores_dict_lasso["Test Score"].append(score_predict)
    

#scores_lasso presents the table of scores metrics of Lasso Model
print('Scores of Lasso Model :\n')
scores_lasso=pd.DataFrame(scores_dict_lasso)
scores_lasso.index=["TotalGHGEmissions","SiteEnergyUse(kBtu)"]
print(scores_lasso)


#Now we need to plot the scores of Lasso Model for Total GHGEmissions and Energy Consumption
fig = plt.figure(figsize=(7, 6))
plt.title('Scores Of Lasso Model',fontsize=18)
plt.bar(scores_lasso.index,scores_lasso['Test Score'],color='#CD5C5C')
plt.ylabel('Scores',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ticks = [20,30,40,50,60,70,80,90,100]
plt.yticks(ticks)
plt.show()


#Now we plot the Comparison of Actual vs Fitted Values For Lasso Model 
plt.figure(figsize=(15,7))
plt.suptitle('Comparison of Actual vs Fitted Values For Lasso Model',fontweight="bold",size=17)#'#32CD32'
plt.subplot(1,2,1)
y_analysis_comparision('TotalGHG Emissions',list(y_test_lasso.values())[0],list(y_pred_lasso.values())[0])
plt.subplot(1,2,2)
y_analysis_comparision('SiteEnergyUse(kBtu)',list(y_test_lasso.values())[1],list(y_pred_lasso.values())[1])










#Ridge Model
#scores_dict_ridge presents our scores metrics for the ridge model
scores_dict_ridge={
        'Train MS Error':[],
        'Train Score':[],
        'Test MS Error':[],
        'Test Score':[]}
#y_test_ridge presents our test data for the ridge model
y_test_ridge={}

#y_test_ridge presents our predictions for the ridge model
y_pred_ridge={}

for column in y_columns.columns:
    
    #Splitting our mdata
    X_train, X_test, y_train, y_test = train_test_split(X,y_columns[column],test_size=0.2,random_state=0)
    
    #scaler presents our scaler for standartization
    scaler=StandardScaler()
    
    # Normalize numerical features in train data using sklearn scaler
    numeric_columns1 = X_train.select_dtypes(['float64', 'int64']).columns
    X_train[numeric_columns1] = scaler.fit_transform(X_train[numeric_columns1])
    
    # Normalize numerical features in test data using sklearn scaler
    numeric_columns2 = X_test.select_dtypes(['float64', 'int64']).columns
    X_test[numeric_columns2] = scaler.fit_transform(X_test[numeric_columns2])
    
    #for preventing errors
    X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    y_train = y_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    y_test = y_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    
    #Model Ridge
    Ridger=Ridge(random_state=0)

    #Fit Ridge model
    Ridger.fit(X_train, y_train)
    #best_ridge=CV_Ridger.best_estimator_
    
    #Train Prediction
    pred_train_ridge = Ridger.predict(X_train)
    
    #Test prediction
    pred_test_ridge = Ridger.predict(X_test)
    
    #append in the dict
    y_test_ridge[column]=y_test
    y_pred_ridge[column]=pred_test_ridge
    
    
    #train_rmse presents the mean squared error for  training
    train_rmse = np.sqrt(mean_squared_error(y_train,pred_train_ridge))
    
    #test_rmse presents the mean squared error for  testing
    test_rmse = np.sqrt(mean_squared_error(y_test,pred_test_ridge))
    
    #score_train presents the score of the training 
    score_train=r2_score(y_train,pred_train_ridge)*100
    
    #score_test presents the score of the testing 
    score_predict=r2_score(y_test,pred_test_ridge)*100
  
 
    #Append in the scores metrics dictionary
    scores_dict_ridge["Train MS Error"].append(train_rmse)
    scores_dict_ridge["Train Score"].append(score_train)
    scores_dict_ridge["Test MS Error"].append(test_rmse)
    scores_dict_ridge["Test Score"].append(score_predict)
    

#scores_ridge presents the table of scores metrics of Ridge Model
print('Scores of Ridge Model :\n')
scores_ridge=pd.DataFrame(scores_dict_ridge)
scores_ridge.index=["TotalGHGEmissions","SiteEnergyUse(kBtu)"]
print(scores_ridge)


#Now we need to plot the scores of Ridge Model for Total GHGEmissions and Energy Consumption
fig = plt.figure(figsize=(7, 6))
plt.title('Scores Of Ridge Model',fontsize=18)
plt.bar(scores_ridge.index,scores_ridge['Test Score'],color='#CD5C5C')
plt.ylabel('Scores',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ticks = [20,30,40,50,60,70,80,90,100]
plt.yticks(ticks)
plt.show()



# Now we will plot the Actual vs Fitted Values for both totalGHGEmissions and SiteEnergyUse(kBtu) for Ridge Model
fig, axs = plt.subplots(1,2,figsize=(15,8))
fig.suptitle('Actual vs Fitted Values for Ridge Model',fontsize=18)

sns.distplot(list(y_test_ridge.values())[0], hist=False, color="r", label="Actual Value",ax=axs[0])
sns.distplot(list(y_pred_ridge.values())[0], color="b",hist=False, label="Fitted Values",ax=axs[0])
axs[0].set_title('TotalGHGEmissions')
axs[0].set_xlabel('TotalGHGEmissions',fontsize=12)
axs[0].set_ylabel('Density',fontsize=12)
axs[0].tick_params(labelcolor='black',labelsize=12)
axs[0].legend(["Actual Value","Fitted Value"])


sns.distplot(list(y_test_ridge.values())[1],hist=False, color="r", label="Actual Value",ax=axs[1])
sns.distplot(list(y_pred_ridge.values())[1],hist=False, color="b", label="Fitted Values",ax=axs[1])
axs[1].set_title('SiteEnergyUse(kBtu)')
axs[1].set_xlabel('SiteEnergyUse(kBtu)',fontsize=12)
axs[1].set_ylabel('Density',fontsize=12)
axs[1].tick_params(labelcolor='black',labelsize=12)
axs[1].legend(["Actual Value","Fitted Value"])

plt.show()









#Linear Regression Model
#scores_lr presents the scores metrics of the Linear Regression Model
scores_dict_lr={
        'Train MS Error':[],
        'Train Score':[],
        'Test MS Error':[],
        'Test Score':[]}
#y_test_lr presents the test data of the Linear Regression Model 
y_test_lr={}

#y_pred_lr presents the predictions of the Linear Regression Model
y_pred_lr={}
for column in y_columns.columns:
    
    #Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X,y_columns[column],test_size=0.2,random_state=200)
    
    #scaler presents our scaler for standartization
    scaler=StandardScaler()
    
    # Normalize numerical features in the train data using sklearn scaler
    numeric_columns1 = X_train.select_dtypes(['float64', 'int64']).columns
    X_train[numeric_columns1] = scaler.fit_transform(X_train[numeric_columns1],y_train)
    
    # Normalize numerical features in the test data using sklearn scaler
    numeric_columns2 = X_test.select_dtypes(['float64', 'int64']).columns
    X_test[numeric_columns2] = scaler.transform(X_test[numeric_columns2])
    
    #for preventing error
    X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    y_train = y_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    y_test = y_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    
    #Linear regression Model
    modellinear = LinearRegression()
    #Fit the model
    modellinear.fit(X_train, y_train)
    
    #Train Prediction
    pred_train_linear= modellinear.predict(X_train)
    
    #Test Prediction
    pred_test_linear= modellinear.predict(X_test)
    
    #Add  the values to dict
    y_test_lr[column]=y_test
    y_pred_lr[column]=pred_test_linear
    
    #train_rmse presents the mean squared error for  training
    train_rmse = np.sqrt(mean_squared_error(y_train,pred_train_linear))
    
    #test_rmse presents the mean squared error for  testing
    test_rmse = np.sqrt(mean_squared_error(y_test,pred_test_linear))
    
    #score_train presents the score of the training 
    score_train=modellinear.score(X_train,y_train)*100
    
    #score_test presents the score of the testing 
    score_predict=modellinear.score(X_test,y_test)*100
    

    #Append in the scores metrics dictionary
    scores_dict_lr["Train MS Error"].append(train_rmse)
    scores_dict_lr["Train Score"].append(score_train)
    scores_dict_lr["Test MS Error"].append(test_rmse)
    scores_dict_lr["Test Score"].append(score_predict)
    
    
#scores_lr presents the table of scores metrics of Linear Regression Model
print('Scores of Linear Regression Model\n:')
scores_lr=pd.DataFrame(scores_dict_lr)
scores_lr.index=["TotalGHGEmissions","SiteEnergyUse(kBtu)"]
print(scores_lr)


#Now we need to plot the scores of Linear Regression Model for Total GHGEmissions and Energy Consumption
fig = plt.figure(figsize=(7, 6))
plt.title('Scores Of Linear Regression Model',fontsize=18)
plt.bar(scores_lr.index,scores_lr['Test Score'],color='#CD5C5C')
plt.ylabel('Scores',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ticks = [20,30,40,50,60,70,80,90,100]
plt.yticks(ticks)
plt.show()


# Now we will plot the Actual vs Fitted Values for both totalGHGEmissions and SiteEnergyUse(kBtu) for Linear Regression Model
fig, axs = plt.subplots(1,2,figsize=(15,8))
fig.suptitle('Actual vs Fitted Values for Linear Regression Model',fontsize=18)

sns.distplot(list(y_test_lr.values())[0], hist=False, color="r", label="Actual Value",ax=axs[0])
sns.distplot(list(y_pred_lr.values())[0], color="b",hist=False, label="Fitted Values",ax=axs[0])
axs[0].set_title('TotalGHGEmissions')
axs[0].set_xlabel('TotalGHGEmissions',fontsize=12)
axs[0].set_ylabel('Density',fontsize=12)
axs[0].tick_params(labelcolor='black',labelsize=12)
axs[0].legend(["Actual Value","Fitted Value"])

sns.distplot(list(y_test_lr.values())[1],hist=False, color="r", label="Actual Value",ax=axs[1])
sns.distplot(list(y_pred_lr.values())[1],hist=False, color="b", label="Fitted Values",ax=axs[1])
axs[1].set_title('SiteEnergyUse(kBtu)')
axs[1].set_xlabel('SiteEnergyUse(kBtu)',fontsize=12)
axs[1].set_ylabel('Density',fontsize=12)
axs[1].tick_params(labelcolor='black',labelsize=12)
axs[1].legend(["Actual Value","Fitted Value"])

plt.show()






#we will build a concatenated dataframe that use concat() function to concatenate only the Test Score column for TotalGHGEmissions

#df_rf_ghg presents the filtered dataframe on the Test Score of the Random Forest Model
df_rf_ghg=scores_rf.loc[['TotalGHGEmissions'] ,['Test Score']].rename_axis('Column')

#df_gb_ghg presents the filtered dataframe on the Test Score of the Gradient Boostig Model
df_gb_ghg=scores_GB.loc[['TotalGHGEmissions'] ,['Test Score']].rename_axis('Column')

#df_lasso_ghg presents the filtered dataframe on the Test Score of the Lasso Model
df_lasso_ghg=scores_lasso.loc[['TotalGHGEmissions'] ,['Test Score']].rename_axis('Column')

#df_ridge_ghg presents the filtered dataframe on the Test Score of the Ridge Model
df_ridge_ghg=scores_ridge.loc[['TotalGHGEmissions'] ,['Test Score']].rename_axis('Column')

#df_linear_ghg presents the filtered dataframe on the Test Score of the Linear Regression Model
df_linear_ghg=scores_lr.loc[['TotalGHGEmissions'] ,['Test Score']].rename_axis('Column')

#scores_totalco2emissions is the concatenated dataframe
scores_totalco2emissions=pd.concat([df_rf_ghg,df_gb_ghg,df_lasso_ghg,df_ridge_ghg,df_linear_ghg])
index_=['Random Forest','Gradient Boosting','Lasso','Ridge','Linear Regression']
scores_totalco2emissions.index=index_
print('Concatenated Dataset of CO2 Emissions:\n',scores_totalco2emissions.reset_index())





#we will build a concatenated dataframe that use concat() function to concatenate only the Test Score column for the SiteEnergyUse(kBtu)

#df_rf_ener presents the filtered dataframe on the Test Score of the Random Forest Model
df_rf_ener=scores_rf.loc[['SiteEnergyUse(kBtu)'] ,['Test Score']].rename_axis('column')

#df_gb_ener presents the filtered dataframe on the Test Score of the Gradient Boosting Model
df_gb_ener=scores_GB.loc[['SiteEnergyUse(kBtu)'] ,['Test Score']].rename_axis('column')

#df_lasso_ener presents the filtered dataframe on the Test Score of the Lasso Model
df_lasso_ener=scores_lasso.loc[['SiteEnergyUse(kBtu)'] ,['Test Score']].rename_axis('column')

#df_ridge_ener presents the filtered dataframe on the Test Score of the Ridge Model
df_ridge_ener=scores_ridge.loc[['SiteEnergyUse(kBtu)'] ,['Test Score']].rename_axis('column')

#df_linear_ener presents the filtered dataframe on the Test Score of the Linear Regression Model
df_linear_ener=scores_lr.loc[['SiteEnergyUse(kBtu)'] ,['Test Score']].rename_axis('column')

#scores_energy is the concatenated dataframe
scores_energy=pd.concat([df_rf_ener,df_gb_ener,df_lasso_ener,df_ridge_ener,df_linear_ener])
index_=['Random Forest','Gradient Boosting','Lasso','Ridge','Linear Regression']
scores_energy.index=index_
print('Conatenated Dataset of energy Consumption:\n',scores_energy.reset_index())





#Comparision of the scores for Energy Comsumption and CO2 emissions
fig, ax= plt.subplots(1,2,figsize=(25,8))
fig.suptitle('Comparision of the scores for Energy Comsumption and CO2 emissions\n',fontweight='bold',fontsize=25)
sns.barplot(x="index", y="Test Score", data=scores_totalco2emissions.reset_index(),ax=ax[0],palette=['#DC143C','#CD5C5C','#F08080','#FA8072','#E9967A'])
ax[0].set_title('TotalGHGEmissions',fontweight='bold',fontsize=20)
ax[0].set_xlabel('Model',fontsize=15)
ax[0].set_ylabel('Test SCore',fontsize=15)
ax[0].tick_params(labelcolor='black',labelsize=10)


sns.barplot(x="index", y="Test Score", data=scores_energy.reset_index(),ax=ax[1],palette=['#DC143C','#CD5C5C','#F08080','#FA8072','#E9967A'])
ax[1].set_title('SiteEnergyUse(kBtu)',fontsize=20,fontweight='bold')
ax[1].set_xlabel('Model',fontsize=15)
ax[1].tick_params(labelcolor='black',labelsize=10)
ax[1].set_ylabel('Test Score',fontsize=15)

plt.show()

