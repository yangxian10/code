#2014-6-10
#deformable structure regression tracking
#by yangxian

import numpy as np
import cv2
import random
import brief

__treeNum = 20
__learnRate = 0.9
__levels = 4
__threshold = 0.3
__delta = 0.02
__fernLenth = 8
__searchRadius = 40

__brief = np.array([])
__label = np.array([])
__hist = np.array([])
__weights = np.array([]) 

def init(inimg, box):
    global __brief
    global __label
    global __hist
    global __weights
    __label = init_label(box)
    __brief = brief_feature(box)
    imgPyr = create_imgPyr(inimg)
    x,y,w,h,label_learn = get_appro_rect(inimg, box, __label)
    learn_win = [x,y,w,h]
    feature = get_feature(imgPyr, box, __brief)
    __hist = cal_conf_hist(__label, feature)
    __weights = np.ones(__treeNum)

def process_frame(inimg, box):
    global __hist
    #detect
    x,y,w,h = get_search_rect(inimg, box)
    search_win = [x,y,w,h]
    imgPyr = create_imgPyr(inimg)
    feature = get_feature(imgPyr, search_win, __brief)
    confmaps = get_conf_map(feature, __hist)
    ymax,xmax = get_max_conf_index(confmaps, __weights)
    xx,yy,ww,hh = trans_tracking_box(inimg, box, search_win, xmax, ymax)
    #learn
    x,y,w,h,learn_label = get_appro_rect(inimg, box, __label)
    learn_win = [x,y,w,h]
    feature = get_feature(imgPyr, learn_win, __brief)
    confmaps = get_conf_map(feature, __hist)
    hist_new = cal_conf_hist(learn_label, feature)
    __hist = update_hist(__hist, hist_new)
    return xx,yy,ww,hh

def init_label(box):
    x,y,w,h = box
    label = np.zeros((h,w))
    out_sigma = np.sqrt(float(w*h)/32.0)
    for i in range(h):
        for j in range(w):
            temp = -0.5/out_sigma/out_sigma*( (i-h/2)**2+(j-w/2)**2 )
            label[i,j] = np.exp(temp)
    return label

def brief_feature(objbox):
    x,y,w,h = objbox
    brief_ftr = np.zeros((__treeNum,__fernLenth,6))
    for i in range(__treeNum):
        for j in range(__fernLenth):
            x1,y1,x2,y2,level1,level2 = brief.create(w,h,__levels)
            brief_ftr[i,j] = [x1,y1,x2,y2,level1,level2]
    return brief_ftr

def get_feature(imgPyr, box, brief_ftr):
    x,y,w,h = box
    feature = np.zeros((__treeNum,h,w))
    weights = 2**np.array(range(__fernLenth-1,-1,-1))
    for i in range(__treeNum):
        featuretemp = np.zeros((h,w))
        for j in range(__fernLenth):
            signimg = brief.getftr(imgPyr, box, brief_ftr[i,j])
            featuretemp += weights[j]*signimg
        feature[i] = featuretemp
    return feature

def cal_conf_hist(label, feature):
    histlenth = range(2**__fernLenth+1)
    hists = np.zeros((__treeNum, len(histlenth)-1))
    for k in range(__treeNum):
        hist, bins = np.histogram(feature[k], histlenth, [feature[k].min(),feature[k].max()], False, label)
        count, bins = np.histogram(feature[k], histlenth)
        hist /= count
        hist[hist!=hist]=0
        hists[k] = hist
    return hists

def update_hist(histold, histnew):
    hist = histold*__learnRate + histnew*(1-__learnRate)
    return hist

def get_conf_map(feature, hists):
    confmaps = np.zeros(feature.shape)
    for k in range(__treeNum):
        for i in range(feature.shape[1]):
            for j in range(feature.shape[2]):
                index = feature[k,i,j]
                confmaps[k,i,j] = hists[k,index]
    return confmaps

def get_max_conf_index(confmaps, weights):
    confmap_sum = np.ones(confmaps[0].shape)
    for i in range(__treeNum):
        confmap_sum = np.multiply(confmap_sum, confmaps[i])
    index = np.argmax(confmap_sum)
    ymax = index/confmap_sum.shape[1]
    xmax = index%confmap_sum.shape[1]
    return ymax,xmax

def trans_tracking_box(img, box, window, xmax, ymax):
    x,y,w,h = box
    wx,wy,ww,wh = window
    x = xmax + wx - w/2
    y = ymax + wy - h/2
    return x,y,w,h

def create_imgPyr(img):
    rows,cols = img.shape
    imgPyr = np.zeros((__levels,rows,cols))
    
    for i in range(__levels):
        scale = 0.5**i
        #sz = (round(rows*scale),round(cols*scale))
        sx = int(round(cols*scale))
        sy = int(round(rows*scale))
        ix = img.shape[1]
        iy = img.shape[0]
        if i==0:
            imgtemp = cv2.resize(img,(sx,sy))
            imgPyr[i] = cv2.resize(imgtemp,(ix,iy))
        else:
            imgtemp = cv2.resize(imgPyr[i-1],(sx,sy))
            imgPyr[i] = cv2.resize(imgtemp,(ix,iy))
    return imgPyr

def get_appro_rect(img, box, label):
    rows,cols = img.shape
    x,y,w,h = box
    wx,wy,ww,wh = box
    if wx < w/2:
        wx = w/2
        ww = x+w-wx
    if wy < h/2:
        wy = h/2
        wh = y+h-wy
    if wx > cols-w*3/2:
        ww = cols-w/2-wx
    if wy > rows-h*3/2:
        wh = rows-h/2-wy
    x1 = wx-x
    y1 = wy-y
    x2 = x1+ww
    y2 = y1+wh
    label_appro = label[y1:y2,x1:x2]
    return wx,wy,ww,wh,label_appro

def get_search_rect(img, box):
    x,y,w,h = box
    rows,cols = img.shape
    cx = x+w/2
    cy = y+h/2
    r = __searchRadius
    x1 = max(w/2, cx-r)
    y1 = max(h/2, cy-r)
    x2 = min(cols-w/2, cx+r)
    y2 = min(rows-h/2, cy+r)
    return x1,y1,x2-x1,y2-y1
