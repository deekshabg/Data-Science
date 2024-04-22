import pandas as pd
import numpy as np
df = pd.read_csv(r'HR_Dataset.csv')
df_copy = df.copy()
df

df.head(5)
df.tail(5)

print("Column Headings:")
print(df.columns)

print("\nStatistical Information:")
print(df.describe())

print("\nDescription:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include='all'))

# Determine the size of the dataset
dataset_size = df.shape

# Display the size of the dataset
print("Size of the dataset:", dataset_size)

#Duplicate Data

duplicate_rows = df.duplicated().sum()
duplicate_rows

null_values = df.isnull().sum()

# Display the null values
print("Null values in the DataFrame:")
print(null_values)

#Data Inconsistencies

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your DataFrame containing the provided features

# Select only the quantitative parameters
quantitative_features = ['EmployeeID', 'Age', 'MonthlyIncome in $', 
                         'NumCompaniesWorked', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                         'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                         'YearsWithCurrManager']

# Create box plots for each quantitative parameter with respect to 'Attrition'
for feature in quantitative_features:
    plt.figure(figsize=(6, 6))
    sns.boxplot(x='Attrition', y=feature, data=df)
    plt.title(f'Box Plot of {feature} with respect to Attrition')
    plt.xlabel('Attrition')
    plt.ylabel(feature)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()

# To impute or remove missing data

#For numerical
#We shall now clean the null values
# Replace null values with mean of the column

# Calculate the mean of the column
mean_value = df['Age'].mean()
# Fill missing values with the mean
df['Age'].fillna(mean_value, inplace=True)

mean_value = df['MonthlyIncome in $'].mean()
df['MonthlyIncome in $'].fillna(mean_value, inplace=True)

mean_value = df['NumCompaniesWorked'].mean()
df['NumCompaniesWorked'].fillna(mean_value, inplace=True)

mean_value = df['TotalWorkingYears'].mean()
df['TotalWorkingYears'].fillna(mean_value, inplace=True)

mean_value = df['TrainingTimesLastYear'].mean()
df['TrainingTimesLastYear'].fillna(mean_value, inplace=True)

mean_value = df['YearsAtCompany'].mean()
df['YearsAtCompany'].fillna(mean_value, inplace=True)

mean_value = df['YearsInCurrentRole'].mean()
df['YearsInCurrentRole'].fillna(mean_value, inplace=True)

mean_value = df['YearsSinceLastPromotion'].mean()
df['YearsSinceLastPromotion'].fillna(mean_value, inplace=True)

mean_value = df['YearsSinceLastPromotion'].mean()
df['YearsSinceLastPromotion'].fillna(mean_value, inplace=True)

mean_value = df['YearsWithCurrManager'].mean()
df['YearsWithCurrManager'].fillna(mean_value, inplace=True)

#For categorical
# Forward fill missing values
df['EmployeeLocation'].fillna(method='ffill', inplace=True)

df['Department'].fillna(method='ffill', inplace=True)

df['Gender'].fillna(method='ffill', inplace=True)

df['MaritalStatus'].fillna(method='ffill', inplace=True)

df['Education'].fillna(method='ffill', inplace=True)

df['JobRole'].fillna(method='ffill', inplace=True)

df['PerformanceRating'].fillna(method='ffill', inplace=True)

df['JobSatisfaction'].fillna(method='ffill', inplace=True)

df['OverTime'].fillna(method='ffill', inplace=True)

#For categorical
# Forward fill missing values
df['EmployeeLocation'].fillna(method='ffill', inplace=True)

df['Department'].fillna(method='ffill', inplace=True)

df['Gender'].fillna(method='ffill', inplace=True)

df['MaritalStatus'].fillna(method='ffill', inplace=True)

df['Education'].fillna(method='ffill', inplace=True)

df['JobRole'].fillna(method='ffill', inplace=True)

df['PerformanceRating'].fillna(method='ffill', inplace=True)

df['JobSatisfaction'].fillna(method='ffill', inplace=True)

df['OverTime'].fillna(method='ffill', inplace=True)

#Data Inconsistencies

# Define a list of special characters to check for
special_characters = ['$' , '&' , '^', '#' , '@']

for column in df.columns:
    # Check if the column contains string values
    if df[column].dtype == 'object':
        # Iterate through each value in the column
        for value in df[column]:
            # Check if the value is a string (avoiding NaN values)
            if isinstance(value, str):
                # Check if the value contains any of the special characters
                for char in special_characters:
                    if char in value:
                        print(f"Special character '{char}' found in column '{column}': '{value}'")
                        break  # No need to check other characters once one is found


data = pd.get_dummies(df,columns=['EmployeeLocation','Department','Gender','MaritalStatus','Education','JobRole','PerformanceRating','JobSatisfaction','OverTime'])
print(data.head())

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Split the data into features Independent Variable/Features (X) and Dependent/Target variable (y)
X = data.drop(['Attrition','EmployeeID'], axis=1)  # Features
y = data['Attrition']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the list of features for scatter plot
quantitative_features = ['EmployeeID', 'Age', 'MonthlyIncome in $', 'NumCompaniesWorked', 
                         'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 
                         'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Create scatter plots for each quantitative feature
for feature in quantitative_features:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature], df['Attrition'], alpha=0.5)
    plt.title(f'Scatter Plot of {feature} vs Attrition')
    plt.xlabel(feature)
    plt.ylabel('Attrition')
    plt.grid(True)
    plt.show()

import seaborn as sns
features = ['EmployeeID', 'EmployeeLocation', 'Age', 'Department', 'Gender', 
            'MaritalStatus', 'Education', 'JobRole', 'JobLevel', 'MonthlyIncome in $', 
            'NumCompaniesWorked', 'TotalWorkingYears', 'TrainingTimesLastYear', 
            'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
            'YearsWithCurrManager', 'Attrition', 'PerformanceRating', 
            'JobSatisfaction', 'OverTime']

# Select relevant columns from the DataFrame
selected_features = ['EmployeeID', 'EmployeeLocation', 'Age', 'Department', 'Gender', 
                     'MaritalStatus', 'Education', 'JobRole', 'JobLevel', 'MonthlyIncome in $', 
                     'NumCompaniesWorked', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                     'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                     'YearsWithCurrManager', 'Attrition', 'PerformanceRating', 
                     'JobSatisfaction', 'OverTime']

# Filter the DataFrame to include only the selected features
df_selected = df[selected_features]

# Create pair plot
sns.pairplot(df_selected, hue='Attrition', diag_kind='kde')
plt.show()

# Calculate the correlation matrix
corr_matrix = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Plot the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='CMRmap_r', fmt=".2f")

# Add title and rotate the x-axis labels
plt.title('Correlation Heatmap of 20 Features')
plt.xticks(rotation=35)

# Show plot
plt.show()

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of 20 Features')
plt.show()

##---------Type the code below this line------------------##
#Mutual Information ( Information Gain)
from sklearn.feature_selection import mutual_info_classif

# Assuming 'df' is your DataFrame containing the 20 features and 'target_variable' is your target variable

# Drop non-numeric columns if present
df_numeric = df.select_dtypes(include=['number'])

# Calculate mutual information scores between features and target variable
mi_scores = mutual_info_classif(df_numeric, df['Attrition'])

# Create a DataFrame to store feature names and their corresponding mutual information scores
mi_df = pd.DataFrame({'Feature': df_numeric.columns, 'MI_Score': mi_scores})

# Sort the DataFrame by mutual information score in descending order
mi_df_sorted = mi_df.sort_values(by='MI_Score', ascending=False)

# Print the top significant features
print("Top Significant Features based on Mutual Information:")
print(mi_df_sorted.head())

#Gini Index
from sklearn.tree import DecisionTreeClassifier

# Assuming 'df' is your DataFrame containing the 20 features including categorical data

# Perform one-hot encoding for categorical features
df_encoded = pd.get_dummies(df.drop('Attrition', axis=1))

# Split data into features and target variable
X = df_encoded
y = df['Attrition']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Extract feature importances based on Gini index
feature_importances = clf.feature_importances_

# Create DataFrame to store feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Gini_Importance': feature_importances})

# Sort features by Gini importance in descending order
top_significant_features = feature_importance_df.sort_values(by='Gini_Importance', ascending=False).head(10)  # Adjust '10' to select top features

# Display top significant features
print("Top significant features based on Gini importance:")
print(top_significant_features)

##---------Type the code below this line------------------##
from sklearn import tree

#Top Feature based on Gini Index
top_features = ['MonthlyIncome in $','TotalWorkingYears','Age','YearsAtCompany','YearsWithCurrManager','YearsInCurrentRole','NumCompaniesWorked','YearsSinceLastPromotion','TrainingTimesLastYear']


#ML Technique 1 - Classification
# Split data into features and target variable
X = df_encoded
y = df['Attrition']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

# Train decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

model = DecisionTreeClassifier(splitter='best', criterion = 'gini')
model.fit(X_train,y_train)

#Plot the decision tree
tree.plot_tree(model)

!pip install pydotplus


from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Assuming 'X' contains your features and 'y' contains the target variable
# Selecting the top features
#top_features = ['MonthlyIncome in $', 'TotalWorkingYears', 'Age', 'YearsAtCompany', 
                #'YearsWithCurrManager', 'YearsInCurrentRole', 'NumCompaniesWorked', 
                ##YearsSinceLastPromotion', 'TrainingTimesLastYear']
X_top_features = X[['MonthlyIncome in $', 'TotalWorkingYears', 'Age', 'YearsAtCompany', 
                    'YearsWithCurrManager', 'YearsInCurrentRole', 'NumCompaniesWorked', 
                    'YearsSinceLastPromotion', 'TrainingTimesLastYear']]
y = df['Attrition']

# Initialize and fit the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_top_features, y)

# Plot the decision tree
plt.figure(figsize=(15, 30))
plot_tree(clf, feature_names=X_top_features.columns, class_names=['No Attrition', 'Attrition'], filled=True)
plt.show()

#Predicting on test data
preds = model.predict(X_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 

pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions

# Accuracy 
np.mean(preds==y_test)

y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
 
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=k_means.classes_, yticklabels=k_means.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title(f"Confusion Matrix\nAccuracy: {accuracy:.2f}")
# plt.show()

precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.2f}')
print(f'F1 Score: {f1:.2f}')
print(recall)

##---------Type the code below this line------------------##

# Clustering

# Normalization function 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df_encoded)

wcss = []
for i in range(1, 8):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 8), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Plotting the centroids of the clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=250, alpha=0.9)
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Visualize the clusters
fig = plt.figure(0)
plt.grid(True)
plt.scatter(X.iloc[:, 0],X.iloc[:, 1], c=y_kmeans, s=20, cmap='viridis')

#Build Cluster algorithm
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
clusters_new = KMeans(4, random_state=42) # No. of Clusters(4) chosen based on the elbow curve
clusters_new.fit(scaled_df)

X = df_encoded
y = df['Attrition']

X_train, X_test,y_train,y_test =  train_test_split(X,y,test_size=0.20,random_state=70)
k_means = KMeans(4, random_state=42)
k_means.fit(X_train)
print(k_means.labels_)
print(y_test)

#bool_list = list(map(bool,X_test))

y_pred = k_means.predict(X_test)
bool_list = list(map(bool,y_pred))
y_test = y_test.map({"Yes":True,"No":False})
score = metrics.accuracy_score(y_test,bool_list)
print('Accuracy:{0:f}'.format(score))

#Assign clusters to the data set
df_encoded['clusterid_new'] = clusters_new.labels_
df['clusterid'] = clusters_new.labels_

df_encoded.groupby('clusterid_new').agg(['mean']).reset_index()

y_pred = k_means.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
 
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=k_means.classes_, yticklabels=k_means.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title(f"Confusion Matrix\nAccuracy: {accuracy:.2f}")
# plt.show()

precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.2f}')
print(f'F1 Score: {f1:.2f}')
print(recall)

