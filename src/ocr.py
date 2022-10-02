# -*- coding:utf-8 -*-
#
import numpy as np
import tensorflow.keras
from scipy import misc
from image import cut_blank
from PIL import Image
#包含的汉字列表（太长了，仅截取了一部分）
hanzi = open('D:/IDEA/Items/OCR/src/4225300126.txt',encoding='utf-8')
hanzi = hanzi.read()[:-1]
dic=dict(zip(range(3062),list(hanzi))) #构建字表

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
#
# batch_size = 128
# nb_classes = 3062
# img_rows, img_cols = 48, 48
# nb_filters = 64
# nb_pool = 2
# nb_conv = 4
#
# model = Sequential()
#
# model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
#                         border_mode='valid',
#                         input_shape=(img_rows, img_cols,1)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
# model.add(Dropout(0.25))
#
# model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

model = tensorflow.keras.models.load_model('D:/IDEA/Items/OCR/src/my_model.hdf5')

import pandas as pd
zy = pd.read_csv('zhuanyi.csv', encoding='utf-8', header=None)
zy.set_index(0, inplace=True)
zy = zy[1]

def viterbi(nodes):
    paths = nodes[0]
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                try:
                    nows[j+i]= paths_[j]*nodes[l][i]*zy[j[-1]+i]
                except:
                    nows[j+i]= paths_[j]*nodes[l][i]*zy[j[-1]+'XX']
            k = np.argmax(nows.values())
            paths[list(nows.keys())[k]] = list(nows.values())[k]
    return list(paths.keys())[np.argmax(paths.values())]

# mode为direact和search
#前者直接给出识别结果，后者给出3个字及其概率（用来动态规划）
def ocr_one(m, mode='direact'):
    m = m[tuple([slice(*i) for i in cut_blank(m)])]#切除最外层白边
    if m.shape[0] >= m.shape[1]: # 把m放入它的外接正方形矩阵
        p = np.zeros((m.shape[0],m.shape[0]))
        p[:,:m.shape[1]] = m
    else:
        p = np.zeros((m.shape[1],m.shape[1]))
        x = (m.shape[1]-m.shape[0])/2
        p[:m.shape[0],:] = m
    # m = misc.imresize(p,(46,46), interp='nearest') #这步和接下来几步，归一化图像为48x48
    im = Image.fromarray(p)
    # size = tuple((np.array(im.size) * 0.99999).astype(int))
    m = np.array(im.resize((46,46),Image.NEAREST))
    p = np.zeros((48, 48))
    p[1:47,1:47] = m #把m放入中心，最外层一圈白边
    m = p
    m = 1.0 * m // m.max()
    # print(m.shape)
    # print(np.array([[m]]).shape)
    k = model.predict(np.array([[m]]).reshape(1,48,48,1), verbose=0)[0]
    ks = k.argsort()#返回值排序前的下标【3，1，2】— return [1,2,0,]
    if mode == 'direact':
        if k[ks[-1]] > 0.5:
            return dic[ks[-1]]
        else:
            return ''
    elif mode == 'search':
        return {dic[ks[-1]]:k[ks[-1]],dic[ks[-2]]:k[ks[-2]],dic[ks[-3]]:k[ks[-3]]}

'''
#直接调用Tesseract
import os
def ocr_one(m):
	misc.imsave('tmp.png', m)
	os.system('tesseract tmp.png tmp -l chi_sim -psm 10')
	s = open('tmp.txt').read()
	os.system('rm tmp.txt \n rm tmp.png')
	return s.strip()
'''
#文本切割：把文字区切割成一个个文字
def cut_line(pl): #mode为direact或viterbi
    # 切除最外围白边
    pl = pl[tuple([slice(*i) for i in cut_blank(pl)])]
    pl0 = pl.sum(axis=0)
    pl0 = np.where(pl0==0)[0]
    #对文字进行切割：统计切割——对单行文字进行垂直方向求和，和为0的列是切割列
                    # t是一个临时遍历，用来存储连续的切割列，并最后算出连续切割列的中间列
    if len(pl0) > 0:
        pl1=[pl0[0]] #第一处切割列
        t=[pl0[0]]
        for i in pl0[1:]:#遍历后面的白边
            if i-pl1[-1] == 1:#连续，放入t
                t.append(i)
                pl1[-1]=i
            else:#不连续 计算t中间列，替换pl1中记录的“连续切割列的最后一列”
                pl1[-1] = sum(t)//len(t) #不连续了就回到中间
                t = [i]
                pl1.append(i) #记录不连续白边的地方
        pl1[-1] = sum(t)//len(t)#这一部是如果连续就用中间列替换最后列，如果不是t中只会右一个数
        pl1 = [0] + pl1 + [pl.shape[1]-1]

        #感觉pl1[i+1]-pl1[i-1]这样会变成两个汉字 ，而不再是单一汉字,于是我自作主张加了个*2
        cut_position = [1.0*(pl1[i+1]-pl1[i-1])//pl.shape[0] > 1.2*2 for i in range(1,len(pl1)-1)]
        cut_position=[pl1[1:-1][i] for i in range(len(pl1)-2) if cut_position[i]] #简单的切割算法
        cut_position = [0] + cut_position + [pl.shape[1]-1]
    else:
        cut_position = [0, pl.shape[1]-1]
    l = len(cut_position)
    for i in range(1, l):#这里应该是均匀切割，但是我没看懂这个流程
        j = int(round(1.0*(cut_position[i]-cut_position[i-1])//pl.shape[0])) #切割出来的条与高的比
        ab = (cut_position[i]-cut_position[i-1])//max(j,1) #ab的意思是每个字多宽
        cut_position = cut_position + [k*ab+cut_position[i-1] for k in range(1, j)] #平均切割
    cut_position.sort()
    return pl, cut_position

def ocr_line(pl, mode='viterbi'): #mode为direact或viterbi
    pl, cut_position = cut_line(pl) #返回文字条 和 切割列
    if mode == 'viterbi':
        text = list(map(lambda i: ocr_one(pl[:,cut_position[i]:cut_position[i+1]+1], mode='search'), range(len(cut_position)-1)))
        return viterbi(text)
    elif mode == 'direact':
        text = list(map(lambda i: ocr_one(pl[:,cut_position[i]:cut_position[i+1]+1]), range(len(cut_position)-1)))
        ''.join(text)