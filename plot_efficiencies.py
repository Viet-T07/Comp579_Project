from matplotlib import pyplot as plt
from os import listdir
import numpy as np
from sklearn import metrics

#Load datas
path = "efficiency_datas/"
curves = []
for i in range(len(listdir(path))):
    curves.append(np.load(path+"learning_curve_"+str(i)+".npy"))


#Calculate Mean
n_curves = len(curves)
mean_curves = np.zeros(len(curves[0]))
for c in curves:
    mean_curves = np.add(mean_curves, c)
mean_curves = np.divide(mean_curves, n_curves)


#Calculate AUC
x = [i for i in range(len(mean_curves))]
print("AUC = ",metrics.auc(x,mean_curves))

#Plot
for curve in curves:
    plt.plot(curve,'red')
plt.plot(mean_curves,'blue')
plt.show()
