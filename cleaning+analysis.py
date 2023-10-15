#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval 
plt.style.use('ggplot')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') 



#Loading data
data_2015 = pd.read_csv("2015energy.csv")
data_2016 = pd.read_csv("2016energy.csv")


#Take a look into our 2 datasets'shapes
print('Shape of 2015 dataset:',data_2015.shape)
print('Shape of 2016 dataset',data_2016.shape)



# taking a quick first look at 2015 data 
print(data_2015.head())


# taking a quick first look at 2016 data 
print(data_2016.head())


#Let us visualize the location column in 2015 dataset
print('Location column in dataset of 2015',data_2015['Location'][0])


#listing columns for both df
col_2015 = data_2015.columns
col_2016 = data_2016.columns


# Looking for difference in columns
def compare_colums(df1,df2):
    columns_1 = list(df1.columns) 
    columns_2 = list(df2.columns)
    same_columns=[]
    diff_columns_2=[]
    diff_columns_1=[]
    
    #Finding missing cols
    for col in columns_2:
        if col in columns_1:
            same_columns.append(col)
        else:
            diff_columns_2.append(col)
    for col in columns_1:
        if col not in columns_2:
            diff_columns_1.append(col)
    return diff_columns_1, diff_columns_2


#The columns present in the 2015 dataset and absent in the 2016 dataset are:
diff_columns_2015, diff_columns_2016 = compare_colums(data_2015,data_2016)
print(diff_columns_2015)


#The columns present in the 2016 dataset and absent in the 2015 dataset are:
diff_columns_2015, diff_columns_2016 = compare_colums(data_2015,data_2016)
print(diff_columns_2016)


# Uniformization of localisation  data in 2015 dataset
print(data_2015['Location'][0])


#We will separate the variable of location column to extract the nested variables

data_2015= pd.concat([data_2015.drop(['Location'], axis=1),data_2015['Location']
                               .map(literal_eval).apply(pd.Series)],axis=1)

#Let's seperate also human adress variable because it consists of a dictionary object
data_2015 = pd.concat([data_2015.drop(['human_address'], axis=1), data_2015['human_address']
                               .map(literal_eval).apply(pd.Series)], axis=1)
print(data_2015.shape)


#We correct the type of latitude, longitude and the zipcode as well as the TaxParcelIdentificationNumbe
data_2015["zip"] = pd.to_numeric(data_2015["zip"])
data_2015["latitude"] = pd.to_numeric(data_2015["latitude"])
data_2015["longitude"] = pd.to_numeric(data_2015["longitude"])


#We havw to rename all the columns in data_2015 to get indentical name with data_2016
data_2015 = data_2015.rename(columns={"latitude":"Latitude", "longitude":"Longitude",
                                      "address":"Address", "city":"City", 
                                      "state":"State", "zip":"ZipCode",'GHGEmissions(MetricTonsCO2e)':'TotalGHGEmissions',
                                     'GHGEmissionsIntensity(kgCO2e/ft2)':'GHGEmissionsIntensity',
                                     'Comment':'Comments'})

#Let's now eliminate the columns of 2015 having no equivalents in 2016
data_2015=data_2015.drop(['OtherFuelUse(kBtu)','2010 Census Tracts',
                'Seattle Police Department Micro Community Policing Plan Areas',
                'City Council Districts','SPD Beats', 'Zip Codes'], axis=1)


#Let's now compare again
diff_columns_2015, diff_columns_2016 = compare_colums(data_2015,data_2016)
print(diff_columns_2015,diff_columns_2016)


#Cleaning Part
#Starting to clean 2015 dataset
#Let's took a look into our 2015 dataset
print(data_2015.info())



#Let's count the how much buidlings we have for each building type
building_type_1 = data_2015['BuildingType'].value_counts().to_frame()
building_type_1
#Let's visualize this building_type variable
fig=building_type_1.plot(kind='bar',color='#1f6997',title='Number of Buildings for every Builiding Type', fontsize=10)
fig.axes.title.set_size(12)
plt.show()


#It is specified in the project that only buildings not intended for housing will be studied.
# Deleting dwelling data
data_2015 = data_2015[~data_2015['BuildingType'].str.contains("Multifamily")]
#We will take a look into our unique variable in Building Type column
print(data_2015['BuildingType'].unique())
print(data_2015.shape)


#Dealing with NaNs
#df_nan the table contating the number of nans in each column
df_nan_2015 = data_2015.isna().sum().sort_values(ascending=False)
print(df_nan_2015)


#Let's visualize the percentage of NaNs in the columns
plt.figure(figsize=(10,10))
plt.title('Proportion of NaN per variable (%)')
sns.barplot(x=df_nan_2015.values/data_2015.shape[0]*100, y=df_nan_2015.index)
plt.show()


#Drop the outlier and comments columns
data_2015.drop(['Outlier','Comments','YearsENERGYSTARCertified'],axis=1,inplace=True)
print(data_2015.shape)


#Elimination of lines containing only NaNs
data_2015.dropna(how = 'all', axis=0, inplace=True)
print(data_2015.shape)


#Fill certain numercial variables by values  with 0
#col_nan_to_zero are the columns which we need to fill with 0
col_nan_to_zero = ['ThirdLargestPropertyUseTypeGFA', 'SecondLargestPropertyUseTypeGFA',
                   'LargestPropertyUseTypeGFA','SteamUse(kBtu)',
                   'Electricity(kBtu)','NaturalGas(kBtu)']
data_2015[col_nan_to_zero] = data_2015[col_nan_to_zero].fillna(0)



#Fill certain categorical variables by values  with No info
#col_no_use are the columns which we need to fill with No info
col_no_use = ['LargestPropertyUseType', 'SecondLargestPropertyUseType',
              'ThirdLargestPropertyUseType']
data_2015[col_no_use] = data_2015[col_no_use].fillna('No information')



# Observations of features
#unique values presents the unique values of our columns
unique_values=[]
columns=[]
for column in data_2015.columns:
    #we need to apped column into columns list
    columns.append(column)
    if data_2015[column].nunique()<20:
        #if the unique vales of the curent column <20, we need to show them and append them into out unique_values list
         unique_values.append(data_2015[column].unique())
    else:
        #otherwise, if our unique value of the current colun >=20, we need toshow its number and then append it into the unique_values list
        unique_values.append(data_2015[column].nunique())
#df_unique present our dataframe that contain 2 columns: one that presents our columns in the 2015 dataset, other presents its unique values
df_uniques=pd.DataFrame({'Column': columns,'Unique Values':unique_values})
#we need to visualise only certain number of columns so we put them into options variabl
options=['NumberofFloors','SiteEUI(kBtu/sf)','SiteEUIWN(kBtu/sf)','SourceEUI(kBtu/sf)','SourceEUIWN(kBtu/sf)','SiteEnergyUse(kBtu)','SiteEnergyUseWN(kBtu)','SteamUse(kBtu)','Electricity(kWh)','Electricity(kBtu)','NaturalGas(therms)','NaturalGas(kBtu)','TotalGHGEmissions','GHGEmissionsIntensity']
#let visualise them 
result1=df_uniques[df_uniques['Column'].isin(options)]
print(result1)



#It's a second method to visualize the columns and its unique values but not through a dataframe
for column in data_2015.columns:
    if data_2015[column].nunique()<20:
        print('Colonne {}, valeurs uniques :\n{}\n'.format(column, data_2015[column].unique()))
    else:
        print('Colonne {}, {} valeurs uniques'.format(column, data_2015[column].nunique()))



#Function for searching of certan strings in 2015 columns.
def search_componant(df, suffix=None):
    componant = []
    for col in df.columns:
        if suffix in col: 
            componant.append(col)
    return componant



#Deleting the variables which have suffixe WN
data_2015.drop(search_componant(data_2015,'WN'), axis=1, inplace=True)



#Deleting the redundant redondantes, we only need to preserve those of kBtu units
redundant_features = ['NaturalGas(therms)','Electricity(kWh)']
data_2015.drop(redundant_features, axis=1, inplace=True)




#Cleaning the Number of Floors column
print('Number of floors unique values:\n',data_2015['NumberofFloors'].unique())
data_2015['NumberofFloors'].fillna(0, inplace=True)
data_2015['NumberofFloors'].replace(0, 1, inplace=True)



data_2015.isnull().sum()



# we have our tagrets values contains one null value, we should remove it

#Cleaning the TotalGHGEmissions column

#Indexes contaning nan
index_with_nan1 = data_2015['TotalGHGEmissions'].index[data_2015['TotalGHGEmissions'].isnull()]

#Dropping the nan indexes
data_2015 = data_2015.drop(index=index_with_nan1)

#Indexes contaning 0
index_a_zero1 = data_2015['TotalGHGEmissions'].index[data_2015['TotalGHGEmissions']==0]

#Dropping indexes containing 0
data_2015 = data_2015.drop(index=index_a_zero1)

#Cleaning 'SiteEnergyUse(kBtu)' column

#Indexes contaning nan
index_with_nan2 = data_2015['SiteEnergyUse(kBtu)'].index[data_2015['SiteEnergyUse(kBtu)'].isnull()]

#Dropping the nan indexes
data_2015 = data_2015.drop(index=index_with_nan2)

#Indexes contaning 0
index_a_zero2 = data_2015['SiteEnergyUse(kBtu)'].index[data_2015['SiteEnergyUse(kBtu)']==0]

#Dropping indexes containing 0
data_2015 = data_2015.drop(index=index_a_zero2)
print(data_2015.shape)



#Let's verify if we have duplicated values in the 2015 dataset
print('Dupilicated Vales in the BuildingID column for the 2015 dataset:',data_2015['OSEBuildingID'].duplicated().sum())
#we don't have



#Let's now take a look into our second dataset 2016
print(data_2016.info())



#Now we will do the same processs for the 2016 dataset since we don't have any difference between this latter and the 2015 dataset
#Starting to clean 2016 dataset
#It is specified in the project that only buildings not intended for housing will be studied.
# Deleting dwelling data
data_2016 = data_2016[~data_2016['BuildingType'].str.contains("Multifamily")]
#We will take a look into our unique variable in Building Type column
print(data_2016['BuildingType'].unique())


#Dealing with NaNs
#df_nan_2016 the table contating the number of nans in each column
df_nan_2016 = data_2016.isna().sum().sort_values(ascending=False)
print(df_nan_2016)


#Let's visualize the percentage of NaNs in the columns
plt.figure(figsize=(10,10))
plt.title('Proportion de NaN par variable (%)')
sns.barplot(x=df_nan_2016.values/data_2016.shape[0]*100, y=df_nan_2016.index)
plt.show()



#Drop the outlier and comments columns
data_2016.drop(['Outlier','Comments','YearsENERGYSTARCertified'],axis=1,inplace=True)
print(data_2016.shape)


#Elimination of lines containing only NaNs
data_2016.dropna(how = 'all', axis=0, inplace=True)
print(data_2016.shape)


#Fill certain numercial variables by values  with 0
#col_nan_to_zero are the columns which we need to fill with 0
col_nan_to_zero = ['ThirdLargestPropertyUseTypeGFA', 'SecondLargestPropertyUseTypeGFA',
                   'LargestPropertyUseTypeGFA','SteamUse(kBtu)',
                   'Electricity(kBtu)','NaturalGas(kBtu)']
data_2016[col_nan_to_zero] = data_2016[col_nan_to_zero].fillna(0)



#Fill certain categorical variables by values  with No info
#col_no_use are the columns which we need to fill with No info
col_no_use = ['LargestPropertyUseType', 'SecondLargestPropertyUseType',
              'ThirdLargestPropertyUseType']
data_2016[col_no_use] = data_2016[col_no_use].fillna('No information')


# Observations of features
for column in data_2016.columns:
    if data_2016[column].nunique()<20:
        print('Colonne {}, valeurs uniques :\n{}\n'.format(column, data_2016[column].unique()))
    else:
        print('Colonne {}, {} valeurs uniques'.format(column, data_2016[column].nunique()))



#Deleting the variables which have suffixe WN
data_2016.drop(search_componant(data_2016,'WN'), axis=1, inplace=True)


#Deleting the redundant values
redundant_features = ['NaturalGas(therms)','Electricity(kWh)']
data_2016.drop(redundant_features, axis=1, inplace=True)



#Cleaning the Neighborhood column, we should update the neighborhood column
data_2016['Neighborhood'].replace('DELRIDGE NEIGHBORHOODS', 'DELRIDGE', inplace=True)
data_2016['Neighborhood']=data_2016['Neighborhood'].map(lambda x: x.upper())



#Displaying the unique values in Neighbohood
print(data_2016['Neighborhood'].unique())



#Replacing the nan values in Compliance Status
data_2016['ComplianceStatus'].replace('Missing Data', np.nan, inplace=True)
data_2016['ComplianceStatus'].replace('Error - Correct Default Data', np.nan, inplace=True)



#Cleaning the Number of Buildings and Number of Floors
data_2016['NumberofBuildings'].fillna(0, inplace=True)
data_2016['NumberofBuildings'].replace(0, 1, inplace=True)
data_2016['NumberofFloors'].fillna(0, inplace=True)
data_2016['NumberofFloors'].replace(0, 1, inplace=True)

#Observation  of null values 
print(data_2016.isnull().sum())


# we have our tagrets values contains one null value, we should remove it

#Cleaning the TotalGHGEmissions column

#Indexes contaning nan
index_with_nan1 = data_2016['TotalGHGEmissions'].index[data_2016['TotalGHGEmissions'].isnull()]

#Dropping the nan indexes
data_2016 = data_2016.drop(index=index_with_nan1)

#Indexes contaning 0
index_a_zero1 = data_2016['TotalGHGEmissions'].index[data_2016['TotalGHGEmissions']==0]

#Dropping indexes containing 0
data_2016 = data_2016.drop(index=index_a_zero1)

#Cleaning 'SiteEnergyUse(kBtu)' column

#Indexes contaning nan
index_with_nan2 = data_2016['SiteEnergyUse(kBtu)'].index[data_2016['SiteEnergyUse(kBtu)'].isnull()]

#Dropping the nan indexes
data_2016 = data_2016.drop(index=index_with_nan2)

#Indexes contaning 0
index_a_zero2 = data_2016['SiteEnergyUse(kBtu)'].index[data_2016['SiteEnergyUse(kBtu)']==0]

#Dropping indexes containing 0
data_2016 = data_2016.drop(index=index_a_zero2)
print(data_2016.shape)


#Let's verify if we have duplicated values in the 2016 dataset
print('Duplicated values in the BuildingID column for 2016 dataset: ',data_2016.duplicated().sum())


#Merging the 2 datasets
#Data is the new concatenated dataset where we will do all the steps of reduction as well as the analysis part
data = pd.concat([data_2015[data_2016.columns],data_2016], axis = 0).sort_values(["DataYear", "OSEBuildingID"])
print(data.shape)



#Let's check the duplicated indexes in the concatenated data
print('Duplicated indexes in the main data: ',data.index.duplicated().sum())



#Making sure that all the duplcated indexes are removed in the main dataset
data = data[~data.index.duplicated()]


#Make sure that all the duplicated indexes are removed in the concatenated data
print('Duplicated indexes in the main data: ',data.index.duplicated().sum())


#Let's check the na values in the dataset
print(data.isnull().sum())


#Strating the Reduction Part
# Deleting variables which do not interest us
#Let's drop the comments column since it's not fundamental to our analysis
data.drop(['State','ZipCode','City'], axis=1, inplace=True)



#Deleting expensive Variables for future years
data = data.drop(['SteamUse(kBtu)','Electricity(kBtu)','NaturalGas(kBtu)'], axis=1)


#columns_to_drop presents the columns we need to drop
columns_to_drop = ['PropertyName', #similar to ID
                   'TaxParcelIdentificationNumber', #similar to ID
                   'ComplianceStatus', #information on the completeness of the data so we have to delete them
                   'DefaultData', #utility of unknown data
                   'Address', #not preserved (latitude and longitude preferred)
                   'ENERGYSTARScore'#similar to SiteEnergyUse(kBtu) 
                   ]
#Areal data
columns_to_drop += ['SiteEUI(kBtu/sf)',
                    'SourceEUI(kBtu/sf)']

for column in columns_to_drop:
    try:
        data.drop(column, axis=1, inplace=True)
    except:
        print('column {} abscent in the dataset'.format(column))





#Analysis Part
#Building_type is a variable that presents the number of unique values in the Building_Type column
building_type = data.groupby(by='BuildingType')['OSEBuildingID'].nunique()
#Font_title presents some characteristics of the pie chart
font_title = {'family': 'serif',
              'color':  '#1d479b',
              'weight': 'bold',
              'size': 18,
             }

fig, ax = plt.subplots(figsize=(6,6))
ax.pie(building_type.values, labels=building_type.index,
       autopct='%1.1f%%', shadow=True, startangle=30,colors=['#85c0f4','#3f9bd9','#6d859d','#737e8a','#777777'],
       textprops=dict(color="black",size=12, weight="bold"))
ax.axis('equal')
ax.set_title("Distribution of building types in the dataset", fontdict=font_title)
plt.show()



#The buildings that are  'NonResidential' represent the major part of the buildings.
#index_to_drop presents the data for "Nonresidential WA" that we need to drop
index_to_drop=data[data['BuildingType']=="Nonresidential WA"].index
data.drop(index_to_drop, inplace=True)



#Let's visualize the istribution of the building year
fig = plt.figure(figsize=(12,8))
ax = sns.histplot(data=data, x='YearBuilt',color='#638cb1', bins=int((data.YearBuilt.max() - data.YearBuilt.min())/5))
ax.set_xlabel("Year of Construction")
ax.set_ylabel("Number of buildings")
plt.title(f"Distribution of Building Year\n", fontdict=font_title)
plt.show()



#we need to add a column that presents the Building Age, this column interest us more than the YearBuilt
data['BuildingAge'] = data['DataYear'] - data['YearBuilt']



#Let's visualize the distribution of the building Age
fig = plt.figure(figsize=(10,7))
ax = sns.histplot(data=data,x='BuildingAge',color='#638cb1',bins=int((data.BuildingAge.max()-data.BuildingAge.min())/5))
ax.set_xlabel("Age of Building")
ax.set_ylabel("Number of Buildings")
plt.title("Distribution of Building Age", fontdict=font_title)
plt.show()


#with the preesence of the 'BuildingAge' variable, we don't want anymore the YearBuilt variable 
data.drop('YearBuilt', axis=1, inplace=True)



#Univariate Analysis
#var_to_hist presents the  categorical features where we will do the univariate analysis  
var_to_hist = ['BuildingType', 'LargestPropertyUseType', 'PrimaryPropertyType','Neighborhood']

for column in var_to_hist:
    fig = plt.figure(figsize=(15,12))
    data[column].value_counts().sort_index(axis=0).plot(kind='bar',color='#1f4e6f')
    plt.title(column,size=20)
    plt.xticks(size = 10)
    plt.yticks(size = 20)
    plt.show()



#BuildingAge 
fig = plt.figure(figsize=(30,12))
data['BuildingAge'].value_counts().sort_index(axis=0).plot(kind='bar',color='#1f4e6f')
plt.title('Building Age',size=20)
plt.xticks(size = 13,weight='bold')
plt.yticks(size = 20,weight='bold')
plt.show()



#PropertyGFATotal
fig = plt.figure(figsize=(5,5))
sns.histplot(data['PropertyGFATotal'],color='#1f4e6f')
plt.title('PropertyGFATotal')
plt.show()



#We will know view the TotalGHGEmissions variable
anaghg=sns.displot(data['TotalGHGEmissions'], palette='bright', height=6, aspect=1.5)
plt.xlim(0,1500)
anaghg.set(title="Distribution of the TotalGHGEmissions variable")
plt.show()



#Kurtosis and decribe of TotaGHGEmissions
print('Kurtosis of Total CO2 Emissions column',data['TotalGHGEmissions'].kurtosis())
print('Skewness of Total CO2 Emissions column',data['TotalGHGEmissions'].skew())
print('Decripstion of Total CO2 Emissions column',data['TotalGHGEmissions'].describe())



#We will know show the distribution of the SiteEnergyUse(kBtu) variable 
anaseu=sns.displot(data['SiteEnergyUse(kBtu)'], palette='bright', height=6, aspect=1.5)
plt.xlim(0,0.5e8)
anaseu.set(title="Distribution of the SiteEnergyUse(kBtu) variable ")
plt.show()



#Kurtosis and decription of SiteEnergyUse(kBtu)
print('Kurtosis of SiteEnergyUse(kBtu) column',data['SiteEnergyUse(kBtu)'].kurtosis())
print('Skewness of SiteEnergyUse(kBtu) column',data['SiteEnergyUse(kBtu)'].skew())
print('Decripstion of SiteEnergyUse(kBtu) column',data['SiteEnergyUse(kBtu)'].describe())




#Bivariate Analysis
#This function allows you to view the distribution 
#of CO2 emissions within the classes of a variable.
def visualisation_CO2(variable,df):
    #mean of TotalGHGEmissions
    the_mean=df["TotalGHGEmissions"].mean()
    fig=plt.figure(figsize=[18,7])
    fig.patch.set_facecolor('#E0E0E0')
    fig.patch.set_alpha(0.7)
    plt.title("C02 emissions distribution by {}".format(variable),size=16)
    sns.boxplot(x=variable, y="TotalGHGEmissions", data=df,color="#cbd1db",width=0.5,showfliers=False,showmeans=True)
    #we need to show th mean value of the TotalGHGEmissions
    plt.hlines(y=the_mean,xmin=-0.5,xmax=len(df[variable].unique())-0.5,color="#6d788b",ls="--",label="Global mean")

    plt.ylabel(" C02 emissions",size=14)
    plt.xticks(range(0,len(df[variable].unique()))
               ,df[variable].unique(),rotation=90)
    plt.legend()
    plt.grid()
    plt.show()
visualisation_CO2('Neighborhood',data)


#This function allows you to view the distribution 
#of Energy  within the classes of a variable.
def visualisation_energy(variable,df):
    #mean of TotalGHGEmissions
    the_mean=df["SiteEnergyUse(kBtu)"].mean()
    fig=plt.figure(figsize=[18,7])
    fig.patch.set_facecolor('#E0E0E0')
    fig.patch.set_alpha(0.7)
    plt.title("Energy distribution by {}".format(variable),size=16)
    sns.boxplot(x=variable, y="SiteEnergyUse(kBtu)", data=df,color="#A9A9A9",width=0.5,showfliers=False,showmeans=True)
    #we need to show th mean value of the TotalGHGEmissions
    plt.hlines(y=the_mean,xmin=-0.5,xmax=len(df[variable].unique())-0.5,color="#6d788b",ls="--",label="Global mean")

    plt.ylabel("Amount of Energy",size=14)
    plt.xticks(range(0,len(df[variable].unique()))
               ,df[variable].unique(),rotation=90)
    plt.legend()
    plt.grid()
    plt.show()
visualisation_energy('Neighborhood',data)



#Distribution of energy consumption and CO2 emissions by type of building
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(20,8))
sns.barplot(x='BuildingType',y='TotalGHGEmissions',data=data, ax=axes[0])
sns.barplot(x='BuildingType',y='SiteEnergyUse(kBtu)',data=data, ax=axes[1])
plt.suptitle("Distribution of energy consumption and CO2 emissions by type of building", fontdict=font_title,size=18)
axes[0].set_xlabel('BuildingType',fontweight='bold')
axes[0].set_ylabel('TotalGHGEmissions',fontweight='bold')
axes[1].set_xlabel('BuildingType',fontweight='bold')
axes[1].set_ylabel('SiteEnergyUse(kBtu)',fontweight='bold')
plt.show()



#Influence of building age on CO2 emissions
fig, axes = plt.subplots( figsize=(30,8))
sns.barplot(data=data,x=data['BuildingAge'], y="TotalGHGEmissions", color = "#9932CC" )
plt.title("Influence of building age on CO2 emissions", fontdict=font_title)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()




#Influence of building age on Energy Consumption
fig, axes = plt.subplots( figsize=(30,8))
sns.barplot(data=data,x=data['BuildingAge'], y="SiteEnergyUse(kBtu)", color = "#4B0082" )
plt.title("Influence of building age on Energy Consumption", fontdict=font_title)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()



#Starting the coorelaions parts
#corr presents the dimesion of the correlation matrix which is equal to the dimension of data.columns
corr = data.corr()
#mask presents the zeros matrx of corr
mask = np.zeros_like(corr)
#this line remove the redundant elements of the matrix since it's a symmetric matrix
mask[np.triu_indices_from(mask)] = True
#starting to show the matrix function using heatmap
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={'size':8}, 
                 mask=mask,center=0,cmap="coolwarm")
plt.title(f"Heatmap of Linear Correlations \n", fontsize = 18)
plt.show()



#we need to view the most correlated feature since this can have a negative effect on the accuracy of models in data mining
#thershold presents the standart coeff_correlation that we put in order to define the strongest correlations
threshold = 0.7

#corr_pairs presents the correlation between each pair of feature
corr_pairs = corr.unstack().sort_values(kind="quicksort")

#strong_corr presents the pair correlation that is greater than 0.7 
strong_corr = (pd.DataFrame(corr_pairs[(abs(corr_pairs) > threshold)]).reset_index().rename(columns={0:'corr_coeff'}))

#we need to take only the the paires that are different, indentical pairs does not interest us
strong_corr = strong_corr[(strong_corr.index%2 == 0) & (strong_corr['level_0'] != strong_corr['level_1'])]

print(strong_corr.sort_values('corr_coeff', ascending=False))



#Note that the suffixed variables GFA show strong correlations with several other variables. 
#We are therefore going to create new variables to try to erase these linear correlations: We are therefore going to create a variable giving us the total number of uses of the building, 
#then delete the complete list of uses:
data['TotalUseTypeNumber'] = [str(word).count(",") + 1 for word in data['ListOfAllPropertyUseTypes'].str.split(',')]
data.drop('ListOfAllPropertyUseTypes', axis=1, inplace=True)



#We are now going to convert the different surfaces (Buildings and Parking) into a percentage of the total surface and we will only keep these 2 variables by deleting the variables LargestPropertyUseTypeGFA, SecondLargestPropertyUseTypeGFA, ThirdLargestPropertyUseTypeGFA
gfa_features = search_componant(data, suffix='GFA')
print(data[['TotalUseTypeNumber'] + gfa_features].head(10))




#Let's create the percentage of the total surface
# Calcul of ratios
data['GFABuildingRate'] = (round((data['PropertyGFABuilding(s)'].fillna(0)
                                  /data['PropertyGFATotal'].fillna(0)),5))
data['GFAParkingRate'] = (round((data['PropertyGFAParking'].fillna(0)
                                 /data['PropertyGFATotal'].fillna(0)),5))

# Removing unnecessary  Data
data.drop(['LargestPropertyUseTypeGFA', 
           'SecondLargestPropertyUseTypeGFA',
           'SecondLargestPropertyUseType',
           'ThirdLargestPropertyUseTypeGFA',
           'ThirdLargestPropertyUseType',
           'PropertyGFAParking',
           'PropertyGFABuilding(s)'],axis=1, inplace=True)

#We complete the uses of the widest part
data['LargestPropertyUseType'] = data['LargestPropertyUseType'].fillna("Unknown")




#We can also calculate the average area per building and per floor:
data['GFAPerBuilding'] = round((data['PropertyGFATotal'] / data['NumberofBuildings']),3)
data['GFAPerFloor'] = round((data['PropertyGFATotal'] / data['NumberofFloors']),3)




#The data is now well completed. 
#We will check the impact of this feature engineering on the linear correlation matrix
corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={'size':8}, 
                 mask=mask, center=0, cmap="coolwarm")
plt.title(f"Heatmap of Linear Correlations\n", fontsize = 18)
plt.show()



#Let's show the shape the our cleaned dataset
print('Shape of the cleaned dataset:',data.shape)


# we will save our cleaned dataset into a new csv file in order to work on the different prediction models
data.set_index("OSEBuildingID").to_csv("cleaned_dataset.csv")

