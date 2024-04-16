import pandas as pd
import numpy as np
from datetime import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

df = pd.read_csv('loan_data_2007_2014.csv')
df_cleaned.info()

selected_columns = ['inq_last_6mths','total_rec_late_fee','delinq_2yrs','loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length',
                    'home_ownership', 'annual_inc', 'verification_status', 'issue_d', 'loan_status', 'purpose',
                    'title', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
                    'initial_list_status', 'application_type', 'total_pymnt','last_pymnt_d','last_credit_pull_d']

df_modified = df[selected_columns]

# Calculate the correlation matrix
correlation_matrix = df_modified.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Show the plot
plt.title("Correlation Matrix Heatmap for Selected Columns")
plt.show()
df_modified.info()

sns.countplot(x='loan_status',data=df_modified)
plt.xticks(rotation=45)
plt.show()

df_modified.loc[:, 'target'] = np.where(
    (df_modified['loan_status'] == 'Charged Off') | 
    (df_modified['loan_status'] == 'Default') | 
    (df_modified['loan_status'] == 'Late (31-120 days)') | 
    (df_modified['loan_status'] == 'Late (16-30 days)') | 
    (df_modified['loan_status'] == 'Does not meet the credit policy. Status:Charged Off'),
    1, 0
)

percentages = df_modified['target'].value_counts(normalize=True) * 100
object_columns = df_modified.select_dtypes(include=['object'])

sns.countplot(x='purpose',data=df_modified)
plt.xticks(rotation=70)
plt.show()

unique_values_count = object_columns.nunique()
unique_values_count

df_modified = df_modified.drop(columns=['emp_title','title','application_type','loan_status'])
columns_with_less_non_null = [col for col in df_modified.columns if df_modified[col].count() < 466285]
print(columns_with_less_non_null)

object_columns = df_modified.select_dtypes(include=['object'])
unique_values_count = object_columns.nunique()

df_modified.groupby('purpose')['loan_amnt'].describe()

df_modified.groupby('target')['loan_amnt'].describe()

correlation_matrix = df_modified.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Show the plot
plt.title("Correlation Matrix Heatmap for Selected Columns")
plt.show()
df_modified.info()

df_modified.groupby('target')['loan_amnt'].describe()

sns.boxplot(x='target', y='loan_amnt', data=df_modified)
sns.boxplot(x='target', y='loan_amnt', data=df_modified)
plt.figure(figsize=(12,4))
subgrade_order = sorted(df_modified['sub_grade'].unique())
sns.countplot(x= 'sub_grade', data=df_modified, order=subgrade_order,hue='target')

percent_missing = df_modified.isnull().sum() * 100 / len(df_modified)
dtypes=[df_modified[col].dtype for col in df_modified.columns]
missing_value_df = pd.DataFrame({'tipe data':dtypes,
                                 'null_percent': percent_missing})
missing_value_df.sort_values('percent_missing', ascending=False, inplace=True)
missing_value_df.head(10)

for col in df_modified.select_dtypes(exclude='object'):
    df_modified[col] = df_modified[col].fillna(df_modified[col].median())
df_modified.isnull().sum()

df_modified = df_modified.dropna(subset=['emp_length'])

df_modified = df_modified.dropna(subset=['earliest_cr_line'])
df_modified = df_modified.dropna(subset=['last_pymnt_d'])
df_modified = df_modified.dropna(subset=['last_credit_pull_d'])
df_modified = df_modified.dropna(subset=['revol_util'])

df_modified[['issue_d','last_pymnt_d','last_credit_pull_d','earliest_cr_line']].info()
from datetime import datetime as dt

# df_modified['issue_d'] = pd.to_datetime(df_modified['issue_d'].apply(lambda x : dt.strptime(x, '%b-%y')))
df_modified['last_pymnt_d'] = pd.to_datetime(df_modified['last_pymnt_d'],format='%b-%y')
df_modified['earliest_cr_line'] =pd.to_datetime(df_modified['earliest_cr_line'],format='%b-%y') 
df_modified['last_credit_pull_d'] = pd.to_datetime(df_modified['last_credit_pull_d'],format='%b-%y') 

df_modified[['issue_d','last_pymnt_d','last_credit_pull_d','earliest_cr_line']].head()
df_modified['start_to_last_year'] = df_modified['last_credit_pull_d'].dt.year - df_modified['earliest_cr_line'].dt.year
df_modified['start_to_last_month'] = (df_modified['last_credit_pull_d'].dt.year - df_modified['earliest_cr_line'].dt.year)*12 + df_modified['last_credit_pull_d'].dt.month - df_modified['earliest_cr_line'].dt.month
df_modified.drop(columns='last_credit_pull_d',inplace=True)
df_modified.drop(columns='earliest_cr_line',inplace=True)

grade_dict = {
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    'E':4,
    'F':5,
    'G':6
}
df_modified['grade'] = df_modified['grade'].map(grade_dict)

df_modified.drop(columns='sub_grade',inplace=True)

df_modified.drop(columns='sub_grade',inplace=True)

home_ownership_dict = {
    'RENT': 'RENT',
    'MORTGAGE' :'MORTGAGE',
    'OWN' : 'OWN',
    'OTHER' :'OTHER',
    'NONE':'OTHER',
    'ANY':'OTHER'
}
df_modified['home_ownership'] = df_modified['home_ownership'].map(home_ownership_dict)

encoder = OneHotEncoder(sparse=False)
df_home_encoded = pd.DataFrame(encoder.fit_transform(df_modified[["home_ownership"]]))
df_home_encoded.columns = encoder.get_feature_names_out(["home_ownership"])
df_modified = df_modified.reset_index(drop=True)
df_modified= pd.concat([df_home_encoded, df_modified],axis=1)
df_modified = df_modified.drop(columns=['home_ownership'])
df_modified['term'] = df_modified['term'].str.extract('(\d+)').astype(int)
df_modified["verification_status"].unique()

encoder = OneHotEncoder(sparse=False)
df_verif_encoded = pd.DataFrame(encoder.fit_transform(df_modified[["verification_status"]]))
df_verif_encoded.columns = encoder.get_feature_names_out(["verification_status"])
df_modified= pd.concat([df_home_encoded, df_modified],axis=1)

df_modified.drop(["verification_status"] ,axis=1, inplace=True)
df_modified.drop(["issue_d"] ,axis=1, inplace=True)


purpose_dict = {
    'credit_card': 'Credit Card',
    'debt_consolidation': 'Debt Consolidation',
    'car': 'Lifestyle and Discretionary Spending',
    'small_business': 'Business and Professional Development',
    'educational': 'Business and Professional Development',
    'home_improvement': 'Personal and Household Needs',
    'house': 'Personal and Household Needs',
    'major_purchase': 'Lifestyle and Discretionary Spending',
    'medical': 'Personal and Household Needs',
    'moving': 'Lifestyle and Discretionary Spending',
    'vacation': 'Lifestyle and Discretionary Spending',
    'wedding': 'Lifestyle and Discretionary Spending',
    'renewable_energy': 'Personal and Household Needs',
    'other': 'Other'
}

df_modified['purpose'] = df_modified['purpose'].map(purpose_dict)

encoder = OneHotEncoder(sparse=False)
df_purpose_encoded = pd.DataFrame(encoder.fit_transform(df_modified[["purpose"]]))
df_purpose_encoded.columns = encoder.get_feature_names_out(["purpose"])
df_modified.drop(["purpose"] ,axis=1, inplace=True)
df_modified= pd.concat([df_purpose_encoded, df_modified],axis=1)

encoder = OneHotEncoder(sparse=False)
df_status_encoded = pd.DataFrame(encoder.fit_transform(df_modified[["initial_list_status"]]))
df_status_encoded.columns = encoder.get_feature_names_out(["initial_list_status"])
df_modified.drop(["initial_list_status"] ,axis=1, inplace=True)
df_modified= pd.concat([df_status_encoded, df_modified],axis=1)

df_modified = df_modified.drop(columns=['last_pymnt_d'])

X = df_modified.drop('target', axis=1).values
y = df_modified['target'].values    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

def eval_classification(model,  y_train_pred, y_test_pred, y_train, y_test):
    print("Accuracy (Train Set): %.2f" % accuracy_score(y_train, y_train_pred))
    print("Accuracy (Test Set): %.2f" % accuracy_score(y_test, y_test_pred))
    print("Precision (Train Set): %.2f" % precision_score(y_train, y_train_pred, zero_division=0))
    print("Precision (Test Set): %.2f" % precision_score(y_test, y_test_pred, zero_division=0))
    print("Recall (Train Set): %.2f" % recall_score(y_train, y_train_pred))
    print("Recall (Test Set): %.2f" % recall_score(y_test, y_test_pred))
    print("F1-Score (Train Set): %.2f" % f1_score(y_train, y_train_pred))
    print("F1-Score (Test Set): %.2f" % f1_score(y_test, y_test_pred))
    
    fpr, tpr, thresholds = roc_curve(y_train, y_train_pred, pos_label=1) # pos_label: label yang kita anggap positive
    print("AUC (Train Set): %.2f" % auc(fpr, tpr))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1) # pos_label: label yang kita anggap positive
    print("AUC (Test Set): %.2f" % auc(fpr, tpr))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def cfm(y_test,y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,6))
    sns.set(font_scale = 1.5)
    ax = sns.heatmap(cf_matrix, annot=True,fmt = 'd')
    plt.title('Confusion Matrix From Test Set',fontsize=18)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    plt.show()

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train,y_train)

y_train_pred = dt.predict(X_train)
y_pred = dt.predict(X_test)

eval_classification(dt, y_train_pred, y_pred, y_train, y_test)
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))