#encoding=utf8
import os
import sys
import csv
import argparse

def cal_roc_points(diffs):
    thres_gap = 0.01
    thresholds = [ 0 + thres_gap * i for i in range(1, int(1 / thres_gap)) ]
    roc_points = []
    for thres in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for d in diffs:
            if d[1] == d[4]:
                exp = 1
            else:
                exp = 0

            if d[-1] < -100:
                continue

            if d[-1] >= thres:
                act = 1
            else:
                act = 0

            if exp == 1 and act == 1:
                tp += 1
            elif exp == 1 and act == 0:
                fn += 1
            elif exp == 0 and act == 0:
                tn += 1
            elif exp == 0 and act == 1:
                fp += 1

        if tn + fp == 0 or tp + fn == 0:
            continue

        FAR = fp * 1.0 / (tn + fp)
        TPR = tp * 1.0 / (tp + fn)
        roc_points.append(['%.5f' % TPR, '%.5f' % FAR, '%.3f' % thres])
    return roc_points

def dump_roc_csv(diffs, res_dir):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    roc_points = cal_roc_points(diffs)
    header = ['TPR', 'FAR', 'Threshold']
    with open(os.path.join(res_dir, "roc_points.csv"), 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(header)
        for d in roc_points:
            wr.writerow(d)

def load_diff(diff_score_file):
    diffs = []
    with open(diff_score_file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        spamreader.next()
        for row in spamreader:
            row[-1] = float(row[-1])

            if row[-1] > 1 and row[-1] > -100:
                row[-1] = row[-1] / 100.0
            diffs.append(row)
    return diffs

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--diff_score_file', type=str, help=u'模型文件路径')
    parser.add_argument('--res_dir', type=str, help=u'测试结果存储目录')

    return parser.parse_args(argv)

def cal(args):
    diffs = load_diff(args.diff_score_file)
    dump_roc_csv(diffs, args.res_dir)

if __name__ == '__main__':
    cal(parse_arguments(sys.argv[1:]))
