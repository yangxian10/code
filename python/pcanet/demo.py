from numpy import *
import time
import PCANet
import pca_net
import MNdata
import linearSVM

train_size = 10000
img_size = 28

# train seperate to trn and val
trn_data = MNdata.train_img[:train_size+1]
trn_labels = MNdata.train_label[:train_size+1]
val_data = MNdata.train_img[train_size:-1]
val_labels = MNdata.train_label[train_size:-1]

test_data = MNdata.test_img
test_labels = MNdata.test_label

#subsampling
trn_data = trn_data[::40]
trn_labels = trn_labels[::40] # around 250
test_data = test_data[::500]
test_labels = test_labels[::500] # around 100

num_test = len(test_labels)

print '====PCANet parameters==='
print PCANet

print '===PCANet training==='

start = time.time()
ftrain, V, blkIdx = pca_net.train(trn_data)
end = time.time()
print 'PCANet training time:', end-start,'s'

print '===training linear svm classifier==='
start = time.time()
#print '---ftrain', shape(ftrain)
#print 'trnlabels', shape(trn_labels)
svm_models = linearSVM.train(ftrain,trn_labels)
end = time.time()
print 'SVM training time:', end-start,'s'

#testing
print '===PCANet testing==='
ncor_recog = 0
rec_history = zeros((num_test,1))
start = time.time()
for i in range(num_test):
    ftest, blkIdx = pca_net.feaExt([test_data[i]],V)
    #print '---ftest', ftest
    label_test = linearSVM.predict(svm_models,ftest)
    groundtruth, = test_labels[i]
    #print 'label:', label_test
    #print 'ground_truth:', groundtruth
    if label_test == groundtruth:
        rec_history[i] = 1
        ncor_recog += 1

end = time.time()
average_time_per_test = (end-start)/num_test
accuracy = float(ncor_recog) / num_test
erRate = 1-accuracy

#result
print '=====result====='
print 'accuracy:', accuracy
print 'test error rate:', erRate
print 'average test time per sample:', average_time_per_test
