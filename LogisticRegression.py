#Muhammad Mustafa Mohsin - 170451340 Syed Raza - 170975760

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path_to_data = "thyroidDF.csv" #<--change this variable
df = pd.read_csv(path_to_data)
df["diagnosis"] = df.apply(lambda row: row.target in 'ABCDEFGH', axis=1)
#convert to numerical values
df['sex'] = df['sex'].fillna(0.5)
df = df.replace('f', 0)
df = df.replace('t', 1)
df = df.replace('F', 0)
df = df.replace('M', 1)

#filter rows
df = df[(df['TSH_measured'] == 1) & (df['T3_measured'] == 1) & (df['TT4_measured'] == 1) & (df['T4U_measured'] == 1) & (df['FTI_measured'] == 1) & (df['on_antithyroid_meds'] == 0) & (df['on_thyroxine'] == 0) & (df['query_on_thyroxine'] == 0) & (df['I131_treatment'] == 0) & (df['age'] < 100)]
# print(df)

#split into independant and dependent variables
all_features = ['age','sex', 'sick', 'pregnant', 'thyroid_surgery', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 
'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'TT4', 'T4U', 'FTI']
x = df[all_features]
y = df.diagnosis 

#feature selection
selector = SelectKBest(mutual_info_classif, k=12)
selector.fit(x, y)
x = x[selector.get_feature_names_out()]

#split into test and train
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#run logistic regression
reg = LogisticRegression(penalty='l1',solver='liblinear', max_iter=1500)
reg.fit(x_train,y_train)
predictions=reg.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, predictions))
print("Precision:",metrics.precision_score(y_test, predictions))
print("Recall:",metrics.recall_score(y_test, predictions))

#ROC curve plot
y_pred_proba = reg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)

#confusion matrix plot
cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual Diagnosis')
plt.xlabel('Predicted Diagnosis')
all_sample_title = 'Accuracy Score: {0}'.format(reg.score(x_test, y_test))
plt.title(all_sample_title, size = 15)

plt.show()
