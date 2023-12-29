import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import plot_tree
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import time
from eiffel2 import builder

df=pd.read_csv('balance-scale.csv')
print(df.head())

df.describe(include='all')
print(df['Class'].value_counts())
sns.countplot(data=df, x='Class', label='count')
x=df.drop('Class',axis=1)
y=df.Class

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3,random_state=30)
clf=DecisionTreeClassifier(criterion="entropy",random_state=30,max_depth=2,min_samples_leaf=3)
clf=clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)

print('Accuracy:',metrics.accuracy_score(y_test, y_pred))

plot_tree(clf)

start=time.time()
neural=MLPClassifier(hidden_layer_sizes=[5,20],activation='relu',alpha=0.001)
neural.fit(x_train,y_train)
predicted=neural.predict(x_test)
print('Accuracy:\n',accuracy_score(y_test,predicted))
end=time.time()
print("Execution time:",end-start,'seconds')


builder([1,10,10,5,5,2], bmode="night")


