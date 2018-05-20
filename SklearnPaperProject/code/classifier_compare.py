#! encoding=utf-8

import json
import random
import crossValidation
from collections import defaultdict
import classifiers

feature_Names = "/Users/lihongsheng/Desktop/MyProject/PaperProject/SklearnPaperProject/data/feature/feature_names.json"
label_file = "/Users/lihongsheng/Desktop/MyProject/PaperProject/SklearnPaperProject/data/label/label_uaip_4.txt"
feature_file = "/Users/lihongsheng/Desktop/MyProject/PaperProject/SklearnPaperProject/data/feature/samples_bfe86_tieba.data.uaip.group.features_no_spurl.json"

def f2json(f):
    with open(f) as fin:
        data = json.load(fin)
    return data

def load_label(label_file):
    result = {}
    for line in open(label_file):
        items = line.rstrip().split()
        key = items[0]
        label = items[1]
        result[key] = int(label)
    return result

def load_feature(feature_file):
    return f2json(feature_file)

def gen_k_idxes(k, ys):
    y_data = defaultdict(list)
    for i, y in enumerate(ys):
        y_data[y].append(i)
    randoms = []
    idxes = [[] for i in range(k)]
    for label, ylist in y_data.items():
        count = len(ylist) / k
        random.shuffle(ylist)
        offset = 0
        for idx_list in idxes:
            idx_list += ylist[offset:offset + count]
            offset += count
        randoms += ylist[offset:]
    random.shuffle(randoms)
    count = len(randoms)/k
    for i, idx_list in enumerate(idxes):
        idx_list += randoms[i*count:i*count+count]
    return idxes

def show_precision(ys, y_preds):
    n = len(ys)
    errors = defaultdict(int)
    error = 0
    ts = 0
    fs = 0
    tp = 0
    fp = 0
    maxy = max(ys)
    y_preds = map(lambda y:min(int(y+0.5), maxy), y_preds)
    classes = [0] * (maxy+1)
    corrects = [0] * (maxy+1)
    recall = [0] * (maxy+1)
    total_correct = 0
    for y, y_pred in zip(ys, y_preds):
        classes[y] += 1
        recall[y_pred] += 1
        if y == y_pred:
            corrects[y] += 1
            total_correct += 1
        if y == 1:
            ts += 1
            if y_pred == 1:
                tp += 1
        else:
            fs += 1
            if y_pred != 1:
                fp += 1
        if y != y_pred:
            error += 1
            errors["%d->%d" % (y, y_pred)] += 1
    newCorreccts = 1
    for index,val in enumerate(recall):
        if newCorreccts == 0:
            newCorreccts = 1
        if val == 0:
            recall[index] = newCorreccts
        newCorreccts = val
        # for debug
        # print index,val
    recalls = map(lambda v:float(v[1])/float(v[0]), zip(classes, corrects)[1:])
    precs = map(lambda v:float(v[1])/float(v[0]), zip(recall, corrects)[1:])
    print "Recalls:", "\t".join(map(lambda v:"%d/%d %.2f" % (v[1], v[0], float(v[1])/v[0]), zip(classes, corrects)[1:]))
    print "Precisions:", "\t".join(map(lambda v:"%d/%d %.2f" % (v[1], v[0], float(v[1])/v[0]), zip(recall, corrects)[1:]))
    print "Total Precision: %d/%d %.2f" % (total_correct, n, float(total_correct)/ n)
    return 1-error/1.0/n, float(fp)/fs, float(tp)/ts, recalls, precs

def validate(xs, ys, clf, clfname):
    idxes = gen_k_idxes(2, ys)
    n = len(ys)
    xgroups = []
    ygroups = []
    results = []
    print "\n", clfname
    idx = 0
    for idx_list in idxes:
        idx += 1
        print "Round:", idx
        test_x = map(lambda i:xs[i], idx_list)
        test_y = map(lambda i:ys[i], idx_list)
        remains = filter(lambda i:not i in idx_list, range(n))
        train_x = map(lambda i:xs[i], remains)
        train_y = map(lambda i:ys[i], remains)
        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)
        prec, tp, fp, recalls, precs= show_precision(test_y, y_pred)

if __name__ == "__main__":
    key2label = load_label(label_file)
    key2features = load_feature(feature_file)
    keys = key2label.keys()
    xs = map(lambda key: key2features[key], keys)
    ys = map(lambda key: key2label[key], keys)
    key2featureNames = load_feature(feature_Names)
    crossValidation.crossValidateMain(xs, ys ,key2featureNames)
    # for clf, clfname in classifiers.clfs:
    #     validate(xs, ys, clf, clfname)