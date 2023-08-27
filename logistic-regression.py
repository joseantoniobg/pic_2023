import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# leitura do arquivo com os dados para análise
dataset = pd.read_csv('dados2.csv')

# assume nossas variáveis de análise como todas as colunas menos a última
x = dataset.iloc[:, :-1].values

# assume nosso resultado 0 ou 1 como a últuma coluna
y = dataset.iloc[:, -1].values

# realiza a reparação dos dados entre os sets de treinamento e de testes, é dito que 25% dos nossos dados serão separados para testar
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

#normaliza os dados. é importante para manter nossos dados numa escala uniforme e remover ruídos
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, :-2] = sc.fit_transform(x_train[:, :-2])
x_test[:, :-2] = sc.transform(x_test[:, :-2])

# print(x_train)
# print(x_test)

# importa e prepara a regressão logística, e realiza o treinamento com o devido set de dados
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# prediction = [[55,1.44,44258,1295.75,1,1]]
# prediction = np.array(prediction)
# prediction[:, :-2] = sc.transform(prediction[:, :-2])

# print(classifier.predict(prediction))

# realiza a predição no set de testes
y_pred = classifier.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import confusion_matrix, accuracy_score

# gera e exibe a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print(cm)

# calcula e exibe a exatidão das predições
print(accuracy_score(y_test, y_pred))

# from matplotlib.colors import ListedColormap
# X_set, y_set = sc.inverse_transform(x_train), y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
#                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
# plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
# plt.title('Logistic Regression (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.savefig('1.png')