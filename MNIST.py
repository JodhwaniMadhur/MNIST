import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


data=pd.read_csv('mnist_train.csv')
#print(data.head)
test_data=pd.read_csv('mnist_test.csv')

train_set=data.head(50000)
validation_set=data.tail(10000)

Y_train=train_set['label']
train_set.drop(labels='label',axis=1,inplace=True)
X_train=train_set


Y_validate=validation_set['label']
validation_set.drop(labels='label',axis=1,inplace=True)
X_validate=validation_set


Y_test=test_data['label']
test_data.drop('label',inplace=True,axis=1)
X_test=test_data

a=ExtraTreesClassifier(n_estimators=500,n_jobs=-1,random_state=42)
a.fit(X_train,Y_train)
ans=a.predict(X_validate)
print("Accuracy of ETC on validation set is: ",100 * accuracy_score(Y_validate,ans))
ans=a.predict(X_test)
print("Accuracy of ETC on test set is: ",100 * accuracy_score(Y_test,ans))


b=RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=42)
b.fit(X_train,Y_train)
ans=b.predict(X_validate)
print("Accuracy of RFC on validation set is: ",100 * accuracy_score(Y_validate,ans))
ans=b.predict(X_test)
print("Accuracy of RFC on test set is: ",100 * accuracy_score(Y_test,ans))


c=LinearSVC(max_iter=100,tol=20,random_state=42)
c.fit(X_train,Y_train)
c.predict(X_validate)
ans=c.predict(X_validate)
print("Accuracy of LinearSVR on validation set is: ",100 * accuracy_score(Y_validate,ans))
ans=c.predict(X_test)
print("Accuracy of LinearSVR on test set is: ",100 * accuracy_score(Y_test,ans))


print(a.score(X_train,Y_train))
print(b.score(X_train,Y_train))
print(c.score(X_train,Y_train))


est=[('extra_trees',a),('random_forest',b),('LinearSVR',c)]
abc=VotingClassifier(est,n_jobs=-1,voting='hard')
abc.fit(X_train,Y_train)
ans=abc.predict(X_validate)
print("Accuracy of Voting on validation set Classifier is: ",100 * accuracy_score(Y_validate,ans))
print(abc.score(X_train,Y_train))
ans=abc.predict(X_test)
print("Accuracy of Voting on test set Classifier is: ",100 * accuracy_score(Y_test,ans))


c=MLPClassifier(random_state=42)
li=[('extra_trees',a),('random_forest',b),('mlp',c)]
abc=VotingClassifier(li,n_jobs=-1)
abc.voting='soft'
abc.fit(X_train,Y_train)
ans=abc.predict(X_validate)
print("Accuracy of (soft)Voting on validation set Classifier is: ",100 * accuracy_score(Y_validate,ans))
print(abc.score(X_train,Y_train))
ans=abc.predict(X_test)
print("Accuracy of (soft)Voting on test set Classifier is: ",100 * accuracy_score(Y_test,ans))


