"""
This projects is about Credit Card Fruad Dectection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data = pd.read_csv(r'C:\Users\Cheney12345\Desktop\Data Mining Pratice Project\CreditCardFraud\creditcard.csv')

count_class = data['Class'].value_counts()

count_class.plot(kind='bar')
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
# the data is totally unbalanced

##########-------------------------------------------------------------------------------###############
"""
This part is to standardize amout data and resampling inbalanced data, and here I use undersample
"""
from sklearn.preprocessing import StandardScaler

data['normalAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

data.drop(['Time', 'Amount'], axis=1, inplace=True)

X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']



number_records_fraud = len(data[data.Class==1])

fraud_indices = np.array(data[data.Class==1].index)

normal_indices = np.array(data[data.Class==0].index)

random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)

under_sample_indicies = np.concatenate([fraud_indices, random_normal_indices])

under_sample_data = data.iloc[under_sample_indicies, :]

print('number of fraud transactions: %.2f'%(len(fraud_indices)/len(under_sample_indicies)))
print('number of normal transactions: %.2f'%(len(random_normal_indices)/len(under_sample_indicies)))
print('Total number of transactions: %d'%len(under_sample_indicies))

######--------------------------------------------------------------------------------------########
"""
This part is to split data into train data and test data
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

print('Number transaction train dataset: %d'%len(X_train))
print('Number transaction test dataset: %d'%len(X_test))
print('Total number of transactions: %d'%(len(X_train)+len(X_test)))

X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']


X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=101)

print('Number transactions train_undersample dataset: %d'%len(X_train_undersample))
print('Number transactions test_undersample dataset: %d'%len(X_test_undersample))
print('Total number of transactions: %d'%(len(X_train_undersample)+len(X_test_undersample)))

#######---------------------------------------------------------------------------------------#########
"""
This part is to get the best parameter C in LogisticRegression
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report


def printing_Kfold_scores(X_train_data,y_train_data):
       
       c_param_range = [0.01, 0.1, 1, 10, 100]
       
       result_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
       
       result_table['C_parameter'] = c_param_range
       
       j = 0
       
       for c_param in c_param_range:
              print('-------------------------')
              print('C_parameter: ', c_param)
              print('-------------------------')
              
              kf = KFold(5,shuffle=False)
              fold = kf.split(y_train_data)
              
              recall_accs = []
              i = 1
              for train_indices, test_indices in fold:
                     lr = LogisticRegression(C=c_param, penalty='l1')
                     lr.fit(X_train_data.iloc[train_indices, :], y_train_data.iloc[train_indices, :].values.ravel())
                     
                     y_pred_undersample = lr.predict(X_train_data.iloc[test_indices, :].values)
                     
                     recall_acc = recall_score(y_train_data.iloc[test_indices, :].values, y_pred_undersample)
                     
                     recall_accs.append(recall_acc)
                     print('Iteration', i,' : recall score = ', recall_acc)
                     i += 1
              
              result_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
              
              j += 1
              print(' ')
              print('Mean recall score', np.mean(recall_accs))
              print(' ')
              
       best_c = result_table.loc[result_table['Mean recall score'].idxmax()]['C_parameter']
       
       print('---------------------------------------------------------------')
       print('Best model to choose from cross validation is with C parameter = ', best_c)
       
       return best_c

best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)
       

############------------------------------------------------------------------------------################
"""
the model with the under_sample_data and using c parameter best_c
"""
lr = LogisticRegression(C=best_c, penalty='l1')

lr.fit(X_train_undersample, y_train_undersample.values.ravel())

y_pred_undersample = lr.predict(X_test_undersample)

cnf_mat = confusion_matrix(y_test_undersample, y_pred_undersample)

cnf_mat_df = pd.DataFrame(cnf_mat)

cnf_mat_df


sns.set_style('whitegrid')
cnf_mat_gra = sns.heatmap(cnf_mat_df, cmap='Blues', annot=True, fmt='g')
cnf_mat_gra.set(xlabel='Predicted label', ylabel='True label', title='Confusion matrix')

print(recall_score(y_test_undersample, y_pred_undersample))

#################-----------------------------------------------------------------------##################

"""
test the whole data with this parameter and model
"""

y_pred_whole = lr.predict(X_test)

cnf_mat_whole = confusion_matrix(y_test, y_pred_whole)

cnf_mat_whole_df = pd.DataFrame(cnf_mat_whole)

cnf_mat_whole_df


sns.set_style('whitegrid')
cnf_mat_gra = sns.heatmap(cnf_mat_whole_df, cmap='Blues', annot=True, fmt='g')
cnf_mat_gra.set(xlabel='Predicted label', ylabel='True label', title='Confusion matrix')

print(recall_score(y_test, y_pred_whole))

###############-----------------------------------------------------------------------######################

"""
plotting ROC curve
"""

##############------------------------------------------------------------------------######################


best_c = printing_Kfold_scores(X_train, y_train)

lr = LogisticRegression(C=best_c, penalty='l1')
lr.fit(X_train, y_train.values.ravel())
y_pred_all = lr.predict(X_test)

cnf_mat_all = confusion_matrix(y_test, y_pred_whole)

cnf_mat_all_df = pd.DataFrame(cnf_mat_whole)

cnf_mat_all_df


sns.set_style('whitegrid')
cnf_mat_gra = sns.heatmap(cnf_mat_all_df, cmap='Blues', annot=True, fmt='g')
cnf_mat_gra.set(xlabel='Predicted label', ylabel='True label', title='Confusion matrix')

print(recall_score(y_test, y_pred_whole))

##########--------------------------------------------------------#################
"""
Set a threshold to find teh most suitable threshold
"""
lr = LogisticRegression(C= 0.01, penalty='l1')

lr.fit(X_train_undersample, y_train_undersample.values.ravel())

y_pred_undersample_proba = lr.predict_proba(X_test_undersample)
# y_pprred = lr.predict(X_test_undersample)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# print(y_pred_undersample_proba)
# print(y_pprred)

cnf_mat_thresholds = []

for i in thresholds:
       y_pred_undersample_proba_threshold = y_pred_undersample_proba[:, 1] > i
       
       cnf_mat_temp = confusion_matrix(y_pred_undersample_proba_threshold, y_test_undersample)
       
       cnf_mat_thresholds.append(cnf_mat_temp)
       
       print(cnf_mat_temp)
       print('Recall Score is :', recall_score(y_test_undersample,y_pred_undersample_proba_threshold))

fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(10, 10))
plt.subplots_adjust(left=0.125, right=0.9, top=0.9,
                wspace=0.2, hspace=0.5)

j = 0
for x in range(0, 3):
       for y in range(0, 3):
              sns.heatmap(cnf_mat_thresholds[j], cmap='Blues', ax=ax[x][y], annot=True,fmt='g')
              ax[x][y].set_title('Threshold >= %.1f'%((j+1)/10), y=1)
              ax[x][y].set_xlabel('True label')
              ax[x][y].set_ylabel('Predicted label')
              j += 1

################----------------------------------------------------############

"""
Ploting ROC curve about above values.
"""

################----------------------------------------------------#############

"""
Following is about SVM and Descion Tree 
"""

from sklearn.svm import SVC

model_svm = SVC()
model_svm.fit(X_train_undersample, y_train_undersample)

print(model_svm.estimator)

from sklearn.metrics import confusion_matrix, classification_report

SVM_pred = model_svm.predict(X_test)
SVM_pred.head()
pd.DataFrame(SVM_pred)[0].value_counts()

print(confusion_matrix(y_test, SVM_pred))
print(classification_report(y_test, SVM_pred))

# >>> print(classification_report(y_test, SVM_pred))
#             precision    recall  f1-score   support

#           0       1.00      0.95      0.97     85299
#           1       0.03      0.92      0.05       144

# avg / total       1.00      0.95      0.97     85443

## from the confusion_matrix and classification_report, the accuracy is not bad


#################-------------------------------------------------##################
"""
This part I am going to find the most suitable parameter to improve data prediction accuracy
like what C and gamma values to use
"""
SVM_param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 

from sklearn.model_selection import GridSearchCV

SVM_grid = GridSearchCV(SVC(),SVM_param_grid,refit=True,verbose=3)

SVM_grid.fit(X_train_undersample,y_train_undersample)

print(SVM_grid.best_params_)

print(SVM_grid.best_estimator_)

SVM_grid_predictions = SVM_grid.predict(X_test)

print(pd.DataFrame(SVM_grid_predictions)[0].value_counts())
y_test['Class'].value_counts()


print(confusion_matrix(SVM_grid_predictions, y_test))
print(classification_report(SVM_grid_predictions, y_test))

##################---------------------------------------------##############################
"""
At this part , I am going to use Decision tree.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
dree_model = DecisionTreeClassifier()

dree_model.fit(X_train_undersample, y_train_undersample)

dree_model_pred = dree_model.predict(X_test)

print(confusion_matrix(y_test, dree_model_pred))
print(classification_report(y_test, dree_model_pred))


##################-------------------------------------------#################################
"""
Below is to visualize tree classifier and random forest
"""
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
# import graphviz
# dot_data = export_graphviz(dree_model, out_file=None) 
# graph = graphviz.Source(dot_data) 
# graph.render("iris")
features = list(X_train.columns[:])
dot_data = StringIO()  
export_graphviz(dree_model, out_file=dot_data,feature_names=features,filled=True,rounded=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
DT_image = Image(graph[0].create_png())
display(DT_image)

## Random Forest##

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train_undersample, y_train_undersample)
rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))





