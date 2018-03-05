#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
from scipy import spatial
import csv
import time
import codecs
import itertools

#path: ${ROOT}/test_type/identity/*.jpg

def get_image_paths(data_dir):
  res = []
  for root, dirs, files in os.walk(data_dir):
    for name in files:
        if name.endswith((".jpg", ".png")):
            res.append(os.path.join(root, name))

  return res


def load_and_align_data(args, image_paths, image_height, image_width, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    failed_paths = []
    success_paths = []

    print(u"开始抓脸， 图片%d张" % len(image_paths))

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = []
    start = time.time()
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if bounding_boxes.shape[0] < 1:
            failed_paths.append(image_paths[i])
            continue

        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_height, image_width), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        success_paths.append(image_paths[i])

    images = np.stack(img_list)
    end = time.time()

    print("抓脸完成，成功抓取 %d, 失败 %d, 耗时 %ds" % (len(img_list), len(failed_paths), (end - start)))
    with open(os.path.join(args.res_dir, u"抓脸失败.txt"), 'wb') as f:
        for p in failed_paths:
            f.write(p)
            f.write('\n')

    return images, success_paths

def get_embeddings(args, image_files):
    emb = None
    images, success_paths = load_and_align_data(args, image_files, args.image_height, args.image_width, args.margin, args.gpu_memory_fraction)
    print(u"开始提取人脸特征， 人脸总数: %d" % len(success_paths))
    start = time.time()
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
    end = time.time()
    print(u"人脸特征提取结束， 耗时 %ds" % (end - start))
    return emb, success_paths

def compute_diff(embs, image_paths, sample_type):
    diffs = []

    for left, right in itertools.combinations(image_paths, 2):
        left_path_comps = left.split('/')
        left_name = left_path_comps[-1]
        left_id = left_path_comps[-2]
        left_test_type = left_path_comps[-3]
        i = image_paths.index(left)

        right_path_comps = right.split('/')
        right_name = right_path_comps[-1]
        right_id = right_path_comps[-2]
        right_test_type = right_path_comps[-3]
        j = image_paths.index(right)

        diff = 1 - spatial.distance.cosine(embs[i,:], embs[j,:])
        is_pos_sample = left_id.strip() == right_id.strip()

        if sample_type == 'pos' and not is_pos_sample:
            continue

        diffs.append([left_test_type, left_id, left_name, right_test_type, right_id, right_name, is_pos_sample, diff])

    return diffs

def dump_csv(diffs, res_dir):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    header = ['左边测试子集', '左边测试对象ID', '左边测试对象图片', '右边测试子集', '右边测试对象ID', '右边测试对象图片', '是否正样本', '相似度']
    with open(os.path.join(res_dir, "diff_score.csv"), 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(header)
        for d in diffs:
            wr.writerow(d)

def start_test(args):
    image_paths = get_image_paths(args.test_images_dir)
    print("total test image count: %d" % len(image_paths))
    embeddings, success_paths = get_embeddings(args, image_paths)
    diffs = compute_diff(embeddings, success_paths, args.test_type)
    print("finish cal diff scores")
    dump_csv(diffs, args.res_dir)
    print("测试完成，结果保存在 %s" % args.res_dir)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help=u'模型文件路径')
    parser.add_argument('--test_images_dir', type=str, help=u'测试图片根目录路径')
    parser.add_argument('--res_dir', type=str, help=u'测试结果存储目录')
    parser.add_argument('--test_type', type=str, help=u'测试类型: pos or neg', default = 'neg')
    parser.add_argument('--image_height', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--image_width', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.5)
    return parser.parse_args(argv)

if __name__ == '__main__':
    start_test(parse_arguments(sys.argv[1:]))
