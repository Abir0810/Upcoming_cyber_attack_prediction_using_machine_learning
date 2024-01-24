#!/usr/bin/env python
# coding: utf-8

# In[130]:


import pandas as pd


# # Friday-WorkingHours-Afternoon-DDos.pcap_ISCX

# In[131]:


df=pd.read_csv(r"E:\AI\Data\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")


# In[132]:


df.head()


# In[133]:


df.info()


# In[134]:


df.isnull().values.any()


# In[135]:


df.isnull().sum().sum()


# In[136]:


df_filled = df.fillna(0)  # Replace NaN with 0


# In[137]:


df_filled.isnull().sum().sum()


# In[138]:


y = df_filled[['label']]
x = df_filled.drop(['label'],axis=1)
x=x.dropna()


# In[139]:


from sklearn.preprocessing import LabelEncoder


# In[140]:


le =LabelEncoder()


# In[141]:


y['label']= le.fit_transform(y['label'])


# In[142]:


y['label'].unique()


# In[143]:


from sklearn.model_selection import train_test_split


# In[144]:


from sklearn.linear_model import LogisticRegression


# In[145]:


logmodel=LogisticRegression()


# In[146]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming the column 'feature' contains your input features and 'target' is the target variable
X = x = df_filled.drop(['label'],axis=1)  # Reshape to a 2D array with one column


# In[147]:


y = df_filled['label'].values
# Make sure the number of rows in X matches the number of elements in y
assert X.shape[0] == len(y), "Number of rows in X must match the number of elements in y"


# In[148]:


# Print the first few rows to inspect the data
print("X:", X[:5])
print("y:", y[:5])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[149]:


X_train.isnull().sum().sum()


# In[150]:


from sklearn.linear_model import LogisticRegression


# In[151]:


logmodel=LogisticRegression()


# In[152]:


logmodel.fit(X_train,y_train)


# In[156]:


predictions = logmodel.predict(X_test)


# In[157]:


from sklearn.metrics import confusion_matrix


# In[158]:


confusion_matrix(y_test,predictions)


# In[163]:


import matplotlib.pyplot as plt


# In[164]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[165]:


cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)


# In[166]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)


# In[167]:


disp.plot()
plt.show()


# In[173]:


from sklearn.metrics import classification_report


# In[175]:


classification_report(y_test,predictions)


# In[177]:


y_predict = logmodel.fit(X_train, y_train).predict(X_test)


# In[178]:


print(classification_report(y_test, y_predict))


# In[183]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)


# In[184]:


neigh.fit(X_train,y_train)


# In[185]:


predictions = logmodel.predict(X_test)


# In[188]:


from sklearn.metrics import confusion_matrix


# In[189]:


confusion_matrix(y_test,predictions)


# In[186]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[187]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[193]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[194]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[195]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[196]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[197]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[198]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX

# In[28]:


import pandas as pd
df=pd.read_csv(r"E:\AI\Data\MachineLearningCVE\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
df.head()


# In[29]:


df.info()
df.isnull().values.any()
df.isnull().sum().sum()
df_filled = df.fillna(0)  # Replace NaN with 0
df_filled.isnull().sum().sum()


# In[30]:


y = df_filled[['label']]
x = df_filled.drop(['label'],axis=1)
x=x.dropna()


# In[31]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y['label']= le.fit_transform(y['label'])
y['label'].unique()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()


# In[32]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming the column 'feature' contains your input features and 'target' is the target variable
X = x = df_filled.drop(['label'],axis=1)  # Reshape to a 2D array with one column
y = df_filled['label'].values
# Make sure the number of rows in X matches the number of elements in y
assert X.shape[0] == len(y), "Number of rows in X must match the number of elements in y"
# Print the first few rows to inspect the data
print("X:", X[:5])
print("y:", y[:5])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.isnull().sum().sum()


# In[33]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[34]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[35]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[39]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[40]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[41]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[42]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[43]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[44]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[45]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[46]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[47]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # Friday-WorkingHours-Morning.pca

# In[48]:


import pandas as pd
df=pd.read_csv(r"E:\AI\Data\MachineLearningCVE\Friday-WorkingHours-Morning.pcap_ISCX.csv")
df.head()


# In[49]:


df.info()
df.isnull().values.any()
df.isnull().sum().sum()
df_filled = df.fillna(0)  # Replace NaN with 0
df_filled.isnull().sum().sum()


# In[50]:


y = df_filled[['label']]
x = df_filled.drop(['label'],axis=1)
x=x.dropna()


# In[51]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y['label']= le.fit_transform(y['label'])
y['label'].unique()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()


# In[52]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming the column 'feature' contains your input features and 'target' is the target variable
X = x = df_filled.drop(['label'],axis=1)  # Reshape to a 2D array with one column
y = df_filled['label'].values
# Make sure the number of rows in X matches the number of elements in y
assert X.shape[0] == len(y), "Number of rows in X must match the number of elements in y"
# Print the first few rows to inspect the data
print("X:", X[:5])
print("y:", y[:5])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.isnull().sum().sum()


# In[53]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[54]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[55]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[56]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[57]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[58]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[59]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[60]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[61]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[62]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[63]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[64]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # Monday-WorkingHours.pcap_ISCX

# In[71]:


import pandas as pd
df=pd.read_csv(r"E:\AI\Data\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv")
df.head()


# In[72]:


df.info()
df.isnull().values.any()
df.isnull().sum().sum()
df_filled = df.fillna(0)  # Replace NaN with 0
df_filled.isnull().sum().sum()


# In[73]:


y = df_filled[['label']]
x = df_filled.drop(['label'],axis=1)
x=x.dropna()


# In[74]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y['label']= le.fit_transform(y['label'])
y['label'].unique()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()


# In[75]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming the column 'feature' contains your input features and 'target' is the target variable
X = x = df_filled.drop(['label'],axis=1)  # Reshape to a 2D array with one column
y = df_filled['label'].values
# Make sure the number of rows in X matches the number of elements in y
assert X.shape[0] == len(y), "Number of rows in X must match the number of elements in y"
# Print the first few rows to inspect the data
print("X:", X[:5])
print("y:", y[:5])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.isnull().sum().sum()


# In[76]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[77]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[78]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[79]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[80]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[81]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[82]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[83]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[84]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[85]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[86]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[87]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX

# In[93]:


import pandas as pd
df=pd.read_csv(r"E:\AI\Data\MachineLearningCVE\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
df.head()


# In[94]:


df.info()
df.isnull().values.any()
df.isnull().sum().sum()
df_filled = df.fillna(0)  # Replace NaN with 0
df_filled.isnull().sum().sum()


# In[95]:


y = df_filled[['label']]
x = df_filled.drop(['label'],axis=1)
x=x.dropna()


# In[96]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y['label']= le.fit_transform(y['label'])
y['label'].unique()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()


# In[97]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming the column 'feature' contains your input features and 'target' is the target variable
X = x = df_filled.drop(['label'],axis=1)  # Reshape to a 2D array with one column
y = df_filled['label'].values
# Make sure the number of rows in X matches the number of elements in y
assert X.shape[0] == len(y), "Number of rows in X must match the number of elements in y"
# Print the first few rows to inspect the data
print("X:", X[:5])
print("y:", y[:5])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.isnull().sum().sum()


# In[98]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[99]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[100]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[101]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[102]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[103]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[104]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[105]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[106]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[107]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[108]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[109]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX

# In[110]:


import pandas as pd
df=pd.read_csv(r"E:\AI\Data\MachineLearningCVE\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df.head()


# In[111]:


df.info()
df.isnull().values.any()
df.isnull().sum().sum()
df_filled = df.fillna(0)  # Replace NaN with 0
df_filled.isnull().sum().sum()


# In[112]:


y = df_filled[['label']]
x = df_filled.drop(['label'],axis=1)
x=x.dropna()


# In[113]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y['label']= le.fit_transform(y['label'])
y['label'].unique()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()


# In[114]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming the column 'feature' contains your input features and 'target' is the target variable
X = x = df_filled.drop(['label'],axis=1)  # Reshape to a 2D array with one column
y = df_filled['label'].values
# Make sure the number of rows in X matches the number of elements in y
assert X.shape[0] == len(y), "Number of rows in X must match the number of elements in y"
# Print the first few rows to inspect the data
print("X:", X[:5])
print("y:", y[:5])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.isnull().sum().sum()


# In[115]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[116]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[117]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[118]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[119]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[120]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[121]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[122]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[123]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[124]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[125]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[126]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # Tuesday-WorkingHours.pcap_ISCX

# In[127]:


import pandas as pd
df=pd.read_csv(r"E:\AI\Data\MachineLearningCVE\Tuesday-WorkingHours.pcap_ISCX.csv")
df.head()


# In[128]:


df.info()
df.isnull().values.any()
df.isnull().sum().sum()
df_filled = df.fillna(0)  # Replace NaN with 0
df_filled.isnull().sum().sum()


# In[129]:


y = df_filled[['label']]
x = df_filled.drop(['label'],axis=1)
x=x.dropna()


# In[130]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y['label']= le.fit_transform(y['label'])
y['label'].unique()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()


# In[131]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming the column 'feature' contains your input features and 'target' is the target variable
X = x = df_filled.drop(['label'],axis=1)  # Reshape to a 2D array with one column
y = df_filled['label'].values
# Make sure the number of rows in X matches the number of elements in y
assert X.shape[0] == len(y), "Number of rows in X must match the number of elements in y"
# Print the first few rows to inspect the data
print("X:", X[:5])
print("y:", y[:5])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.isnull().sum().sum()


# In[132]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[133]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[134]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[135]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[136]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[137]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[138]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[139]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[140]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[141]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[142]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[143]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX

# In[1]:


import pandas as pd
df=pd.read_csv(r"E:\AI\Data\MachineLearningCVE\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df.head()


# In[2]:


df.info()
df.isnull().values.any()
df.isnull().sum().sum()
df_filled = df.fillna(0)  # Replace NaN with 0
df_filled.isnull().sum().sum()


# In[3]:


y = df_filled[['label']]
x = df_filled.drop(['label'],axis=1)
x=x.dropna()


# In[4]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y['label']= le.fit_transform(y['label'])
y['label'].unique()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming the column 'feature' contains your input features and 'target' is the target variable
X = x = df_filled.drop(['label'],axis=1)  # Reshape to a 2D array with one column
y = df_filled['label'].values
# Make sure the number of rows in X matches the number of elements in y
assert X.shape[0] == len(y), "Number of rows in X must match the number of elements in y"
# Print the first few rows to inspect the data
print("X:", X[:5])
print("y:", y[:5])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.isnull().sum().sum()


# In[6]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[7]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[8]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[9]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[10]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[11]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[12]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[13]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[14]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[15]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[16]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[17]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # Wednesday-workingHours.pcap_ISCX

# In[18]:


import pandas as pd
df=pd.read_csv(r"E:\AI\Data\MachineLearningCVE\Wednesday-workingHours.pcap_ISCX.csv")
df.head()


# In[19]:


df.info()
df.isnull().values.any()
df.isnull().sum().sum()
df_filled = df.fillna(0)  # Replace NaN with 0
df_filled.isnull().sum().sum()


# In[20]:


y = df_filled[['label']]
x = df_filled.drop(['label'],axis=1)
x=x.dropna()


# In[21]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
y['label']= le.fit_transform(y['label'])
y['label'].unique()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()


# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming the column 'feature' contains your input features and 'target' is the target variable
X = x = df_filled.drop(['label'],axis=1)  # Reshape to a 2D array with one column
y = df_filled['label'].values
# Make sure the number of rows in X matches the number of elements in y
assert X.shape[0] == len(y), "Number of rows in X must match the number of elements in y"
# Print the first few rows to inspect the data
print("X:", X[:5])
print("y:", y[:5])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.isnull().sum().sum()


# In[23]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[24]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[25]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[27]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[28]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[29]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[30]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[31]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# In[32]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[33]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[34]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)
y_predict = logmodel.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # The END

# In[ ]:




