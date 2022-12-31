import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import io

#If you want run in google colab, you can download dataset by this code: 
#! wget https://github.com/m-ebrahimzadeh/Farsi-Digit-Classification/raw/e3b36fde338cab84c3a605110add516af1710f28/Data_hoda_full.mat -P dataset

#load data from matlab format 
dataset = io.loadmat('/content/dataset/Data_hoda_full.mat')

print( type(dataset), '\n',
       dataset.keys(), '\n',
       dataset['Data'].shape )

#remove redundent axis
data = np.squeeze(dataset['Data'], axis=1)
lbl = np.squeeze(dataset['labels'])
print(data.shape, lbl.shape)

#show a sample
sample_num = 90
print('This image shows number ', lbl[sample_num], '\n')
plt.imshow(data[sample_num], cmap='gray')
plt.show()

#select data from dataset
x_train = data[:55000]
y_train = lbl[:55000]

x_test = data[55000:]
y_test = lbl[55000:]
print(x_train.shape, '\n', x_test.shape)

#resize selected data
new_dimention = 10
resized_x_train = [cv2.resize(img, (new_dimention, new_dimention)) for img in x_train]
resized_x_test = [cv2.resize(img, (new_dimention, new_dimention)) for img in x_test]

#show a changed sample
plt.imshow(resized_x_train[sample_num], cmap='gray')
plt.show()

#reshape data to a vector
X_train = np.reshape(resized_x_train, [-1, new_dimention**2])
X_test = np.reshape(resized_x_test, [-1, new_dimention**2])

print(X_train.shape, '\n', X_test.shape)

#build model
from sklearn.neighbors import KNeighborsClassifier as KNN
model = KNN(n_neighbors=3, weights='distance')

#train
model.fit(X_train, y_train)

#evaluate
print(model.score(X_test, y_test) * 100, '%')