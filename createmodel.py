# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img

#import  Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
#
import glob
import os
import time
import datetime
from PIL import Image,ImageOps
from google.colab import files

print(tf.__version__)
print(tf.test.gpu_device_name())
now = datetime.datetime.now()+ datetime.timedelta(hours=9)

X = []
Y = []

# BADの画像#
images0 = glob.glob(os.path.join('/content/bad', "*.jpg"))

#images0= files.upload()

targetsize=(128,128)

for i in range(len(images0)):
    img = img_to_array((load_img(images0[i], grayscale=False, target_size=targetsize)))
    img2 = cv2.flip(img, 0)
    img3 = cv2.flip(img, 1)
    img4 = cv2.flip(img, 2)
    img5 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img6 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    img7 = cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)
    img8 = cv2.rotate(img4, cv2.ROTATE_90_CLOCKWISE)
    X.append(img_to_array(img))
    X.append(img_to_array(img2))
    X.append(img_to_array(img3))
    X.append(img_to_array(img4))
    X.append(img_to_array(img5))
    X.append(img_to_array(img6))
    X.append(img_to_array(img7))
    X.append(img_to_array(img8))
    Y.extend([0, 0, 0, 0, 0, 0, 0, 0])


print("1/3 BAD Load" ,i,  len(X))

# Goodの画像
images1 = glob.glob(os.path.join('/content/good', "*.jpg"))
for i in range(len(images1)):
    img = img_to_array(load_img(images1[i], grayscale=False, target_size=targetsize))
    img2 = cv2.flip(img, 0)
    img3 = cv2.flip(img, 1)
    img4 = cv2.flip(img, 2)
    img5 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img6 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    img7 = cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)
    img8 = cv2.rotate(img4, cv2.ROTATE_90_CLOCKWISE)
    X.append(img_to_array(img))
    X.append(img_to_array(img2))
    X.append(img_to_array(img3))
    X.append(img_to_array(img4))
    X.append(img_to_array(img5))
    X.append(img_to_array(img6))
    X.append(img_to_array(img7))
    X.append(img_to_array(img8))
    Y.extend([1, 1, 1, 1, 1, 1, 1, 1])

print("2/3 Good Load", i, len(X))



# Doubleの画像
images4 = glob.glob(os.path.join('/content/double', "*.jpg"))
for i in range(len(images4)):
    img = img_to_array(load_img(images4[i], grayscale=False, target_size=targetsize))
    img2 = cv2.flip(img, 0)
    img3 = cv2.flip(img, 1)
    img4 = cv2.flip(img, 2)
    img5 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img6 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    img7 = cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)
    img8 = cv2.rotate(img4, cv2.ROTATE_90_CLOCKWISE)
    X.append(img_to_array(img))
    X.append(img_to_array(img2))
    X.append(img_to_array(img3))
    X.append(img_to_array(img4))
    X.append(img_to_array(img5))
    X.append(img_to_array(img6))
    X.append(img_to_array(img7))
    X.append(img_to_array(img8))
    Y.extend([4, 4, 4, 4, 4, 4, 4, 4])
print("3/3 Double Load",i, len(X))



# arrayに変換
X = np.asarray(X)
Y = np.asarray(Y)

# クラスの形式を変換


# 学習用データとテストデータ
train_images, test_images, train_labels, test_labels = train_test_split(X, Y, test_size=0.15, random_state=111)

#/Users/Toshi/PycharmProjects/TEST0001/BeanTRAIN

class_names = ['OMOTE_Bad', 'OMOTE_Good', 'URA_Bad', 'URA_Good', 'Double']

#train_images = train_images.astype('float32')

train_images = (train_images / 255.0 *0.99)+0.01
test_images = (test_images / 255.0 *0.990 )+0.01

print(test_images.shape)

train_images = train_images.reshape(train_images.shape[0], 128, 128, 3)
test_images = test_images.reshape(test_images.shape[0], 128, 128, 3)

# One-hot encode the labels
train_labels = tf.keras.utils.to_categorical(train_labels, 5)
test_labels = tf.keras.utils.to_categorical(test_labels, 5)
#x_test = x_test.reshape(x_test.shape[0],*image_shape)
print("Start Learning")
start = time.time()


#通常のモデル作成　Compile the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(128, 128, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=5))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.5))

#model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

#MobileNET
#model = tf.keras.applications.MobileNet(input_shape=(64, 64, 3), alpha=1.0, depth_multiplier=1, weights=None, classes=5)
#model = tf.keras.applications.MobileNetV2(input_shape=(64, 64, 3), alpha=1.0, weights=None, classes=5)

#callback設定
savefilename =  now.strftime('%Y%m%d_%H%M')+'model_output.h5' 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ファイル名！！！！
callbacklist= [tf.keras.callbacks.EarlyStopping(monitor='val_acc',patience=30),
               tf.keras.callbacks.ModelCheckpoint(filepath=savefilename,monitor='val_loss',mode='min',verbose=1,save_best_only=True, ),]
               #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=10 ,)]

model.compile(loss='binary_crossentropy', optimizer='rmsprop',  metrics=['accuracy'])

tf.global_variables_initializer()
history = model.fit(train_images, train_labels, batch_size=32, epochs=200,callbacks=callbacklist, validation_data= (test_images,test_labels))

#model.save(savefilename)    #■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■　　ファイル名変更！！！

print(model.summary())

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")




#　Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
acc= history.history['acc']
val_acc=history.history['val_acc']
loss= history.history['loss']
val_loss=history.history['val_loss']
epochs = range(1, len(loss)+1)

# グラフの表示
plt.plot(epochs, acc, 'bo', label ='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs, loss, 'ro', label ='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()



