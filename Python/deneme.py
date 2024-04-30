# Gerekli kütüphanelerin içe aktarılması
import numpy as np 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns



# X ve y değerlerinin oluşturulması
x = np.arange(100).reshape(-1, 1)
y = np.array([0]*50 + [1]*50)




# Verisetinin train ve test olarak ayrılması
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size =.3)



# Modelin eğitilmesi
model = LogisticRegression().fit(X_train,y_train)


y_pred = model.predict(X_test)



# Accuracy score değerinin hesaplanması
from sklearn.metrics import accuracy_score

print("Accucarcy score: ", accuracy_score(y_test,y_pred))


# Karmaşıklık matrisinin ısı haritası olarak çizilmesi
from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.show()