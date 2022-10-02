import numpy as np
from scipy import misc, ndimage
# import cv2
import imageio
from PIL import Image
from scipy.stats import gaussian_kde as kde
from tqdm import *
import matplotlib.pyplot as plt
# def color2gray(color_img):
#     size_h, size_w, channel = color_img.shape
#     gray_img = np.zeros((size_h, size_w), dtype=np.uint8)
#     for i in range(size_h):
#         for j in range(size_w):
#             gray_img[i, j] = round((color_img[i, j, 0]*11 + color_img[i, j, 1]*59 + \
#                                     color_img[i, j, 2]*30)/100)
#     return gray_img

def myread(filename):
    print('读取图片中...')
    pic = imageio.imread(filename,as_gray=True)
    #三种读取图片的方法结果不同
    # ①
    # pic =cv2.imread(filename)
    #  pic= cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    # ②
    # pic =Image.open(filename)
    # pic = pic.convert('L')

    # # 插值放大图片
    pic = ndimage.zoom(pic, 2)
    # 插值导致区分度下降，使用幂次变换,并映射到【0，255】
    pic = pic ** 2
    pic = ((pic - pic.min()) / (pic.max() - pic.min()) * 255).round()
    print('读取完成...')
    return pic


def decompose(pic):  # 核密度聚类，给出极大值、极小值点、背景颜色、聚类图
    print('图层聚类分解中...')
    d0 = kde(pic.reshape(-1), bw_method=0.2)(range(256))
    d = np.diff(d0)
    d1 = np.where((d[:-1] < 0) * (d[1:] > 0))[0]  # 极小值
    d1 = [0] + list(d1) + [256]
    d2 = np.where((d[:-1] > 0) * (d[1:] < 0))[0]  # 极大值
    if d1[1] < d2[0]:
        d2 = [0] + list(d2)
    if d1[len(d1) - 2] > d2[len(d2) - 1]:
        d2 = list(d2) + [255]

    dc = sum(list(map(lambda i: d2[i] * (pic >= d1[i]) * (pic < d1[i + 1]), range(len(d2)))))
    print('分解完成. 共%s个图层' % len(d2))
    return dc


def erosion_test(dc):  # 抗腐蚀能力测试
    print('抗腐蚀能力测试中...')
    layers = []
    # bg = np.argmax(np.bincount(dc.reshape(-1)))
    # d = [i for i in np.unique(dc) if i != bg]
    d = np.unique(dc)
    for k in d:
        f = dc == k
        label_im, nb_labels = ndimage.label(f, structure=np.ones((3, 3))) # 划分连通区域

        ff = ndimage.binary_erosion(f)  # 腐蚀操作
        def test_one(i):
            index = label_im == i
            if (1.0 * ff[index].sum() / f[index].sum() > 0.9) or (1.0 * ff[index].sum() / f[index].sum() < 0.1):
                f[index] = False

        ff = list(map(test_one, trange(1, nb_labels + 1)))

        layers.append(f)
    print('抗腐蚀能力检测完毕.')
    return layers

def pooling(layers): #以模仿池化的形式整合特征
    print('整合分解的特征中...')
    result = sum(layers)
    label_im, nb_labels = ndimage.label(result, structure=np.ones((3,3)))

    def pool_one(i):
        index = label_im==i
        k = np.argmax([1.0*layers[j][index].sum()/result[index].sum() for j in range(len(layers))])
        result[index] = layers[k][index]
    t = list(map(pool_one, trange(1, nb_labels+1)))
    print('特征整合成功.')
    return result


def post_do(pic):


    label_im, nb_labels = ndimage.label(pic, structure=np.ones((3,3)))
    print('图像的后期去噪中...')
    def post_do_one(i):
        index = label_im==i
        index2 = ndimage.find_objects(index)[0]#外切矩阵
        ss = 1.0 * len(pic.reshape(-1))/len(pic[index2].reshape(-1))**2
        #先判断是否低/高密度区，然后再判断是否孤立区。
        if (index.sum()*ss < 16) or ((1+len(pic[index2].reshape(-1))-index.sum())*ss < 16):
            pic[index] = False
        else:
            a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
            index3 = (slice(max(0, 2*a-b),min(pic.shape[0], 2*b-a)), slice(max(0, 2*c-d),min(pic.shape[1], 2*d-c)))
            if (pic[index3].sum() == index.sum()) and (1.0*index.sum()/(b-a)/(d-c) > 0.75):
                pic[index2] = False
    t = list(map(post_do_one, trange(1, nb_labels+1)))
    print('后期去噪完成.')
    return pic

def areas(pic): #圈出候选区域
    print('正在生成候选区域...')
    pic_ = pic.copy()
    label_im, nb_labels = ndimage.label(pic_, structure=np.ones((3,3)))
    def areas_one(i):
        index = label_im==i
        index2 = ndimage.find_objects(index)[0]
        pic_[index2] = True
    t = list(map(areas_one, trange(1, nb_labels+1)))
    return pic_

#定义距离函数，返回值是距离和方向
#注意distance(o1, o2)与distance(o2, o1)的结果是不一致的
def distance(o1, o2):
    delta = np.array(o2[0])-np.array(o1[0])
    d = np.abs(delta)-np.array([(o1[1]+o2[1])/2.0, (o1[2]+o2[2])/2.0])
    d = np.sum(((d >= 0)*d)**2)
    theta = np.angle(delta[0]+delta[1]*1j)
    k = 1
    if np.abs(theta) <= np.pi/4:
        k = 4
    elif np.abs(theta) >= np.pi*3/4:
        k = 2
    elif np.pi/4 < theta < np.pi*3/4:
        k = 1
    else:
        k = 3
    return d, k

def integrate(pic, k=0): #k=0是全向膨胀，k=1仅仅水平膨胀
    label_im, nb_labels = ndimage.label(pic, structure=np.ones((3,3)))
    def integrate_one(i):
        index = label_im==i
        index2 = ndimage.find_objects(index)[0]
        a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
        cc = ((a+b)/2.0,(c+d)/2.0)
        return (cc, b-a, d-c)
    print('正在确定区域属性...')
    A = list(map(integrate_one, trange(1, nb_labels+1)))
    print('区域属性已经确定，正在整合邻近区域...')
    aa,bb = pic.shape
    pic_ = pic.copy()
    def areas_one(i):
        dist = [distance(A[i-1], A[j-1]) for j in range(1, nb_labels+1) if i != j]
        dist = np.array(dist)
        ext = dist[np.argsort(dist[:,0])[0]] #通过排序找最小，得到最邻近区域
        if ext[0] <= (min(A[i-1][1],A[i-1][2])/4)**2:
            ext = int(ext[1])
            index = label_im==i
            index2 = ndimage.find_objects(index)[0]
            a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
            if ext == 1: #根据方向来膨胀
                pic_[a:b, c:min(d+(d-c)//4,bb)] = True
            elif ext == 3:
                pic_[a:b, max(c-(d-c)//4,0):d] = True
            elif ext == 4 and k == 0:
                pic_[a:min(b+(b-a)//6,aa), c:d] = True #基于横向排版假设，横向膨胀要大于竖向膨胀
            elif k == 0:
                pic_[max(a-(b-a)//6,0):b, c:d] = True
    t = list(map(areas_one, trange(1, nb_labels+1)))
    print('整合完成.')
    return pic_



def cut_blank(pic): #切除图片周围的白边，返回范围，显然是只能切除最外围白边
    try:
        # 竖向列求和
        q = pic.sum(axis=1)
        #找到不含空白列的区域，（左，右边界）
        ii,jj = np.where(q!= 0)[0][[0,-1]]
        xi = (ii, jj+1)
        #横向行求和
        q = pic.sum(axis=0)
        ii,jj = np.where(q!= 0)[0][[0,-1]]
        yi = (ii, jj+1)
        # 返回区域
        return [xi, yi]
    except:
        return [(0,1),(0,1)]

def trim(pic, pic_, prange=5): #剪除白边，删除太小的区域
    label_im, nb_labels = ndimage.label(pic_, structure=np.ones((3,3)))
    def trim_one(i):
        index = label_im==i
        index2 = ndimage.find_objects(index)[0]
        box = (pic*index)[index2]
        [(a1,b1), (c1,d1)] = cut_blank(box)
        pic_[index] = False
        if (b1-a1 < prange) or (d1-c1 < prange) or ((b1-a1)*(d1-c1) < prange**2): #删除小区域
            pass
        else: #恢复剪除白边后的区域
            a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
            pic_[a+a1:a+b1,c+c1:c+d1] = True
    t = map(trim_one, trange(1, nb_labels+1))
    return pic_

def bound(m):
    frange = (slice(m.shape[0]-1), slice(m.shape[1]-1))
    f0 = np.abs(np.diff(m, axis=0))
    f1 = np.abs(np.diff(m, axis=1))
    f2 = np.abs(m[frange]-m[1:,1:])
    f3 = f0[frange]+f1[frange]+f2[frange] != 0
    return f3

def trim_bound(pic, pic_): #剪除白边，删除太小的区域
    pic_ = pic_.copy()
    label_im, nb_labels = ndimage.label(pic_, structure=np.ones((3,3)))
    def trim_one(i):
        index = label_im==i
        index2 = ndimage.find_objects(index)[0]
        box = pic[index2]
        if 1.0 * bound(box).sum()/box.sum() < 0.15:
            pic_[index] = False
    t = map(trim_one, trange(1, nb_labels+1))
    return pic_

pic = myread('pic2.jpeg')
plt.imshow(pic)
plt.show()
dc= decompose(pic)
layers =erosion_test(dc)
plt.imshow(layers[1])
plt.show()
result = pooling(layers)
pic = post_do(result)
pic_ = areas(pic)

plt.imshow(pic_)
plt.show()

pic_ = integrate(pic_,1)
plt.imshow(pic_)
plt.show()

pic_=trim(pic,pic_)
plt.imshow(pic_)
plt.show()

pic_ = integrate(pic_,1)
plt.imshow(pic_)
plt.show()

pic_=trim(pic,pic_,10)
plt.imshow(pic_)
plt.show()

pic_ = trim_bound(pic,pic_)
plt.imshow(pic_)
plt.show()
# _------------------------------------
# plt.imshow(dc)
# plt.title('decompose')
# plt.show()
#
# plt.imshow(layers[4])
# plt.title('erosion_test')
# plt.show()
#
# plt.imshow(result)
# plt.title('pooling')
# plt.show()
#
# plt.imshow(pic)
# plt.title('post_do')
# plt.show()
#
# plt.imshow(pic_)
# plt.show()
