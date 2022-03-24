import cv2
import glob
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D , Flatten, MaxPool2D, BatchNormalization, Dropout, GlobalMaxPool2D
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils import shuffle
#Saving Image Matrix into pickles

fromPath = 'C:/Users/Raj/Desktop/Nensi/test/'
className = {0:'Aavad',1:'Chikoo',2:'Jamun',3:'Raat_Rani',4:'Umbaro'}

'''li = []
labels =[]
for key,values in className.items() :
    for img in glob.glob(fromPath + values + '/*.jpg'):
        cv_img = cv2.imread(img)
        print(img)
        b, g, r = cv2.split(cv_img)
        # cv2.imshow('image',g)
        g = cv2.resize(g,(224,224))
        g = g.reshape(g.shape[0], g.shape[1], 1)
        print('g: ',g.shape)
        li.append(g)
        labels.append(key)


features = 'test'+".pkl"
class_labels = 'TestClassLabels'+".pkl"

li = np.array(li)
labels = np.array(labels)

fo = open(features, "wb")
pickle.dump(li, fo)
fo.close()

fo = open(class_labels, "wb")
pickle.dump(labels, fo)
fo.close()

print(labels)
print(li.shape)'''


#Training of CNN model

fp = open('train.pkl', "rb")
train_features = pickle.load(fp)
fp.close()

fp = open('TrainClassLabels.pkl', "rb")
train_cls_labels = pickle.load(fp)
fp.close()

fp = open('test.pkl', "rb")
test_features = pickle.load(fp)
fp.close()

fp = open('TestClassLabels.pkl', "rb")
test_cls_labels = pickle.load(fp)
fp.close()

X_train = train_features/255
X_test = test_features/255
Y_train = train_cls_labels
Y_test = test_cls_labels
X_train, Y_train = shuffle(X_train, Y_train)
X_test, Y_test = shuffle(X_test, Y_test)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(learning_rate=0.001),metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=50,validation_data=(X_test,Y_test))
model.save('model.h5')

Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred,axis=1)
acc = accuracy_score(Y_test, Y_pred)
print('testing accuracy: ',acc)
