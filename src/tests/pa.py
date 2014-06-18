"""
Created on Dec 22, 2012

@author: Alexandre

The goal of this test is to verifiy that the PUAdapter really allows a regular estimator to
achieve better accuracy in the case where the \"negative\" examples are contaminated with a
number of positive examples.

Here we use the breast cancer dataset from UCI. We purposely take a few malignant examples and
assign them the bening label and consider the bening examples as being \"unlabled\". We then compare
the performance of the estimator while using the PUAdapter and without using the PUAdapter. To
asses the performance, we use the F1 score, precision and recall.

Results show that PUAdapter greatly increases the performance of an estimator in the case where
the negative examples are contaminated with positive examples. We call this situation positive and
unlabled learning.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../')
from puLearning.puAdapter import PUAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


def load_breast_cancer(path):
    f = open(path)
    lines = f.readlines()
    f.close()

    examples = []
    labels = []

    for l in lines:
        spt = l.split(',')
        label = float(spt[-1])
        feat = spt[:-1]
        if '?' not in spt:
            examples.append(feat)
            labels.append(label)

    return np.array(examples), np.array(labels)

import csv
import random
if __name__ == '__main__':
    np.random.seed(42)

    print "Loading dataset"
    print
    label_file=csv.reader(open('../datasets/train_target_first.csv','r'))
    data_file=csv.reader(open('../datasets/data_first.csv','r'))

    label_file.next()
    data_file.next()

    positive_labels=[]
    for id,target in label_file:
        positive_labels.append(id)
    print len(positive_labels)
    features=[]
    i=0
    positive_features=[]
    negative_features=[]
    test_features=[]
    test_ids=[]
    charlist=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    orlist=['']
    for line in data_file:
        rawfeature=line[1:]
        feature=[]
        for item in rawfeature:
            if item=='':
                feature.append(-1.0)
            elif item in charlist:
                feature.append(float(ord(item)))
            else:
                feature.append(float(item))
        if line[0] in positive_labels:
            positive_features.append(feature)
            i+=1
        else:
            if random.random()<0.5:
                test_features.append(feature)
                test_ids.append(line[0])

            # output.write(line[0])
        #     output.write('\n')
        #     i+=1
        if i==100000:
            break
        # if i==20:
        #     break
        # if i==111192:
        #     break

    plabels=[1.0 for item in positive_features]
    print "PU learning in progress..."
    estimator = RandomForestClassifier(n_estimators=100,
                                       criterion='gini',
                                       bootstrap=True,
                                       n_jobs=1)
    pu_estimator = PUAdapter(estimator)
    print positive_features[0]
    print len(positive_features)
    print len(plabels)
    pu_estimator.fit(np.copy(positive_features), np.copy(plabels))
    nlabels = pu_estimator.predict(np.copy(test_features))
    outputids=test_ids[nlabels==1]
    predict_file='data_predict.txt'
    output=open(predict_file,'w')
    for id in outputids:
        output.write(id)
        output.write('\n')
    output.close()


