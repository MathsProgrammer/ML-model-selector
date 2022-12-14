import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree  import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score,precision_score,accuracy_score,recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier


#Read and show data

data = pd.read_csv('diabetes.csv')
data.head()
data.describe().round(2)


#Check for null or duplicated, drop duplicates

data.isnull().sum()  
data.duplicated().sum() 
data = data.drop_duplicates() 

#X,y test/train

X = data.drop('Outcome', axis = 1)
y = data['Outcome']


#Training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=(2))

#Gaussian naive Bayes

classifier=GaussianNB()
classifier.fit(X_train , y_train)
classifier.score(X_test , y_test).round(5)
naive_preds = classifier.predict(X_test)

#Decision tree

dt =DecisionTreeClassifier(max_features=8 , max_depth=6)
dt.fit(X_train , y_train)
dt.score(X_test , y_test).round(5)
DT_preds = dt.predict(X_test)

#Random forest

Rclf = RandomForestClassifier(max_features=8 , max_depth=6)
Rclf.fit(X_train, y_train)
Rclf.score(X_test, y_test).round(5)
Rclf_preds = Rclf.predict(X_test)

#XGBoost

xgb = XGBClassifier()
xgb.fit(X_train , y_train)
xgb.score(X_test , y_test).round(5)
xgb_preds=xgb.predict(X_test)

#AdaBoost

ada = AdaBoostClassifier()
ada.fit(X_train , y_train)
ada.score(X_test , y_test).round(5)
ada_preds=ada.predict(X_test)


#Need to scale the data for the following methods

scale = StandardScaler()
X_train, X_test = scale.fit_transform(X_train),scale.fit_transform(X_test)
    
#Support vector machines

svm = svm.SVC()    
svm.fit(X_train , y_train)
svm.score(X_train , y_train).round(5)
svm_preds = svm.predict(X_test)


#Logistic regression

lr = LogisticRegression(C = 100)
lr.fit(X_train , y_train)
lr.score(X_test , y_test).round(5)
LR_preds = lr.predict(X_test)

#K-Nearest-Neighbours

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn.score(X_test, y_test).round(5)
knn_preds = knn.predict(X_test)

#Neural networks

net = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
net.fit(X_train, y_train)
net.score(X_test, y_test).round(5)
net_preds = net.predict(X_test)


#Results

model_comparison={}
model_comparison['Decision Tree']=[accuracy_score(y_test,DT_preds),f1_score(y_test,DT_preds,average='weighted'),precision_score(y_test,DT_preds),recall_score(y_test,DT_preds)]
model_comparison['Naive Bayes']=[accuracy_score(y_test,naive_preds),f1_score(y_test,naive_preds,average='weighted'),precision_score(y_test,naive_preds),recall_score(y_test,naive_preds)]                                   
model_comparison['Random forest']=[accuracy_score(y_test,Rclf_preds),f1_score(y_test,Rclf_preds,average='weighted'),precision_score(y_test,Rclf_preds),recall_score(y_test,Rclf_preds)]
model_comparison['KNN']=[accuracy_score(y_test,knn_preds),f1_score(y_test,knn_preds,average='weighted'),precision_score(y_test,knn_preds),recall_score(y_test,knn_preds)] 
model_comparison['XGboost']=[accuracy_score(y_test,xgb_preds),f1_score(y_test,xgb_preds,average='weighted'),precision_score(y_test,xgb_preds),recall_score(y_test,xgb_preds)]
model_comparison['Logistic Regerssion']=[accuracy_score(y_test,LR_preds),f1_score(y_test,LR_preds,average='weighted'),precision_score(y_test,LR_preds),recall_score(y_test,LR_preds)] 
model_comparison['Adaboost']=[accuracy_score(y_test,ada_preds),f1_score(y_test,ada_preds,average='weighted'),precision_score(y_test,ada_preds),recall_score(y_test,ada_preds)]
model_comparison['SVM']=[accuracy_score(y_test,svm_preds),f1_score(y_test,svm_preds,average='weighted'),precision_score(y_test,svm_preds),recall_score(y_test,svm_preds)]
model_comparison['Neural networks']=[accuracy_score(y_test,net_preds),f1_score(y_test,net_preds,average='weighted'),precision_score(y_test,net_preds),recall_score(y_test,net_preds)]


df = pd.DataFrame(model_comparison).T
df.columns=['Model Accuracy','Model F1-Score','precision','recall']
df = df.sort_values(by='Model Accuracy',ascending=False)

print(df)
