#2014-6-10
#deformable structure regression tracking
#by yangxian

import random
import numpy as np

def create(w,h,levels):
    x1 = random.randint(-w/2,w/2-1)
    y1 = random.randint(-h/2,h/2-1)
    x2 = random.randint(-w/2,w/2-1)
    y2 = random.randint(-h/2,h/2-1)
    level1 = random.randint(0,levels-1)
    level2 = random.randint(0,levels-1)
    return x1,y1,x2,y2,level1,level2

def getftr(imgpyr, box, brief_ftr):
    x,y,w,h = box
    x1,y1,x2,y2,lv1,lv2 = brief_ftr
    img1 = imgpyr[lv1,y+y1:y+y1+h,x+x1:x+x1+w]
    img2 = imgpyr[lv2,y+y2:y+y2+h,x+x2:x+x2+w]
    outimg = np.sign(img1-img2)
    outimg[outimg<=0]=0
    return outimg
