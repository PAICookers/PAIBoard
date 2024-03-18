import pickle
import numpy as np
import os
# import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def CreatData(dataset_root):
    testpath=os.path.join(dataset_root,'test_batch')
    test_dict=unpickle(testpath)
    testdata=test_dict[b'data'].astype('float32')
    testlabels=np.array(test_dict[b'labels'])
    
    return testdata,testlabels


if __name__ == "__main__":
    dataset_root = '/home/anne/work/99_datasets/cifar-10-batches-py'
    testdata,testlabels = CreatData(dataset_root)

    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataDict = testdata[0]
    print(testlabels[0], classes[testlabels[0]])
    # dataDict = dataDict.reshape(3,32,32)
    # dataDict = np.transpose(dataDict,(1,2,0))
    # max_n = dataDict.max()
    # min_n = dataDict.min()
    # cv_img = 255*(dataDict-min_n)/(max_n-min_n)
    # cv_img = np.asarray(cv_img.astype(np.uint8), order="C")
    # # cv2.imwrite('1.jpg', cv_img)

    # img = cv2.resize(cv_img,(500,500))
    # cv2.imwrite('1.jpg', img)
    # cv2.imshow(classes[testlabels[0]], img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()