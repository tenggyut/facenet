#encoding=utf8
import tensorflow as tf

def am_logits_compute(embeddings, label_batch, fc_dim, nrof_classes):

    '''
    loss head proposed in paper:<Additive Margin Softmax for Face Verification>
    link: https://arxiv.org/abs/1801.05599

    embeddings : normalized embedding layer of Facenet, it's normalized value of output of resface
    label_batch : ground truth label of current training batch
    args:         arguments from cmd line
    nrof_classes: number of classes
    '''
    m = 0.35
    s = 30

    with tf.name_scope('AM_logits'):
        kernel = tf.Variable(tf.truncated_normal([fc_dim, nrof_classes]))
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
        cos_theta = tf.matmul(embeddings, kernel_norm)#(batch_size, nrof_classes) 表征了每个feature与对应权重的夹角
        cos_theta = tf.clip_by_value(cos_theta, -1,1)
        phi = cos_theta - m
        label_onehot = tf.one_hot(label_batch, nrof_classes)
        adjust_theta = s * tf.where(tf.equal(label_onehot,1), phi, cos_theta)

        return adjust_theta
