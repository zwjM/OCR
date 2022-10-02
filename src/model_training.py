# -*- coding:utf-8 -*-
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print('exist!')
    gpu0 = gpus[0]
    tf.config.experimental.set_memory_growth(gpu0,True)
    tf.config.set_visible_devices([gpu0],'GPU')

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import pandas as pd
import glob
import matplotlib.pyplot as plt

hanzi = open('4225300126.txt',encoding='utf-8')
hanzi = hanzi.read()[:-1]

#生成文字矩阵
def gen_img(text, size=(48,48), fontname='simhei.ttf', fontsize=48):
    im = Image.new('1', size, 1)
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(fontname, fontsize)# ImageFont实例
    dr.text((0, 0), text, font=font)
    return (((np.array(im.getdata()).reshape(size)==0)+(np.random.random(size)<0.05)) != 0).astype(float)

#生成训练样本
data = pd.DataFrame()
fonts = glob.glob('./*.[tT][tT]*')

for fontname in fonts:
    print(fontname)
    for i in range(-2,3):
        m = pd.DataFrame(pd.Series(list(hanzi),dtype='str').apply(lambda s:[gen_img(s, fontname=fontname, fontsize=48+i)]))
        m['label'] = range(3062)
        data = data.append(m, ignore_index=True)
        m = pd.DataFrame(pd.Series(list(hanzi),dtype='str').apply(lambda s:[gen_img(s, fontname=fontname, fontsize=48+i)]))
        m['label'] = range(3062)
        data = data.append(m, ignore_index=True)

x = np.array(list(data[0])).astype(float)
np.save('x', x) #保存训练数据

dic=dict(zip(range(3062),list(hanzi))) #构建字表
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras.utils import np_utils

batch_size = 256
nb_classes = 3062
nb_epoch = 30

img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 4

x = np.load('x.npy')            #汉字数*字体*5种字号*每种字号两张照片
y = np_utils.to_categorical(list(range(3062))*2*5*2, nb_classes)
weight = ((3062-np.arange(3062))/3062.0+1)**3
weight = dict(zip(list(range(3063)),weight/weight.mean())) #调整权重，高频字优先

x = x.reshape(x.shape[0], img_rows, img_cols, 1)

model = Sequential()

model.add(Convolution2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x, y,
                    batch_size=batch_size,epochs=nb_epoch,
                    class_weight=weight)

score = model.evaluate(x,y)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save('my_model.hdf5')
model.save_weights('model1.hdf5')