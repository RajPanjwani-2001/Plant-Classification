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


#Data Augmentation
fromPath = '../test/'
augPath = '../aug_test/'
className = {0:'Aavad',1:'Chikoo',2:'Jamun',3:'Raat_Rani',4:'Umbaro'}


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

for key,values in className.items() :
    for image in glob.glob(fromPath + values + '/*.jpg'):
        img = cv2.imread(image)
        # cv2.imshow('image',g)
        da_img = img.reshape(1,img.shape[0], img.shape[1], 3)
        print('new img shape: ',da_img.shape)
        i=0
        for batch in datagen.flow(da_img,save_to_dir=augPath + values,save_format='jpg'):
            i += 1
            if i>20:
                break

#Saving Image Matrix into pickles
li = []
labels =[]
for key,values in className.items() :
    for img in glob.glob(augPath + values + '/*.jpg'):
        g = cv2.imread(img)
        print(g.shape)
        # cv2.imshow('image',g)
        g = cv2.resize(g,(224,224))
        g = g.reshape(g.shape[0], g.shape[1], 3)
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
print(li.shape)
'''



gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=4500)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

#Loading pickle files
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

#Normalizng data
X_train = train_features/255
X_test = test_features/255
Y_train = train_cls_labels
Y_test = test_cls_labels
X_train, Y_train = shuffle(X_train, Y_train)
X_test, Y_test = shuffle(X_test, Y_test)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)

#Training of CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(learning_rate=0.01),metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=10,validation_data=(X_test,Y_test))
model.save('model.h5')

#Checking accuracy
Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred,axis=1)
acc = accuracy_score(Y_test, Y_pred)
print('testing accuracy: ',acc)
'''