import struct
import numpy

dataPath = 'MNIST\\'

def loadImg(filename) :
    
#    imageFileName = dataPath + 'train-images.idx3-ubyte'
#    imageFileName = dataPath + 'train-labels.idx1-ubyte'

    imageFileName = dataPath + filename
    fileData = open(imageFileName, 'rb')
    
    fileData.read(4)
    imageNum, = struct.unpack('>l', fileData.read(4))
    rows, = struct.unpack('>l', fileData.read(4))
    cols, = struct.unpack('>l', fileData.read(4))
    
    sizebuf = rows * cols
    img = []
    for i in range(imageNum):
        Img = struct.unpack('>784B',fileData.read(sizebuf))
        Img = numpy.mat(Img)
        img.append(Img.reshape(rows,cols))
    return img

def loadLabel(filename) :

    labelFileNmae = dataPath + filename
    fileData = open(labelFileNmae, 'rb')

    fileData.read(4)
    labelNum, = struct.unpack('>l', fileData.read(4))
    labels = []
    for i in range(labelNum):
        label = struct.unpack('>B',fileData.read(1))
        labels.append(label)
    return labels

#def loadMNIST():
train_img_file = 'train-images.idx3-ubyte'
train_label_file = 'train-labels.idx1-ubyte'
test_img_file = 't10k-images.idx3-ubyte'
test_label_file = 't10k-labels.idx1-ubyte'

print 'MNIST data loading...'

train_img = loadImg(train_img_file)
train_label = loadLabel(train_label_file)
test_img = loadImg(test_img_file)
test_label = loadLabel(test_label_file)

print 'MNIST data loaded.'
    
        
    

