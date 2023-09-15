from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import sys
import time
import math
import argparse
import data_tools

TRAIN_DIR = './origin_model/checkpoint/'
BATCH_SIZE = 200
MAX_STEPS = 20000
import csv

def write_metrics_to_csv(file_path, metrics):
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = [
            '步骤', '训练完成轮数','训练损失', '测试损失',
            '训练准确率', '测试准确率',
            '训练真正例', '训练真负例', '训练假正例', '训练假负例',
            '测试真正例', '测试真负例', '测试假正例', '测试假负例',
            '训练Precision', '训练Recall', '训练F1值',
            '测试Precision', '测试Recall', '测试F1值'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric)
def inference_graph(input, input_size, keep_prob):
    # NN structure parameters
    L1_patch_w = 4
    L1_patch_h = 1
    L1_depth = 10
    L2_patch_w = 5
    L2_patch_h = 1
    L2_depth = 20
    L3_patch_w = 4
    L3_patch_h = 1
    L3_depth = 20
    L4_patch_w = 4
    L4_patch_h = 1
    L4_depth = 40
    fc_size = 128
    fc_input_size = (input_size - L1_patch_w - L2_patch_w - 4 * L3_patch_w - 4 * L4_patch_w + 25) // 16
    
    x_input = tf.reshape(input, [-1,input_size,1,1])
    
    # Convolutional 1
    with tf.name_scope('conv1'):
        weights = tf.Variable(tf.truncated_normal([L1_patch_w,L1_patch_h,1,L1_depth], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[L1_depth]), name='biases')
        conv1 = tf.nn.relu(tf.nn.conv2d(x_input, weights, strides=[1, 1, 1, 1], padding='VALID') + biases)
    # Convolutional 2
    with tf.name_scope('conv2'):
        weights = tf.Variable(tf.truncated_normal([L2_patch_w,L2_patch_h,L1_depth,L2_depth], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[L2_depth]), name='biases')
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights, strides=[1, 2, 1, 1], padding='SAME') + biases)
    # Pooling 1
    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(conv2, ksize=[1,2,1,1], strides=[1,2,1,1], padding='VALID')
    # Convolutional 3
    with tf.name_scope('conv3'):
        weights = tf.Variable(tf.truncated_normal([L3_patch_w,L3_patch_h,L2_depth,L3_depth], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[L3_depth]), name='biases')
        conv3 = tf.nn.relu(tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='VALID') + biases)
    # Convolutional 4
    with tf.name_scope('conv4'):
        weights = tf.Variable(tf.truncated_normal([L4_patch_w,L4_patch_h,L3_depth,L4_depth], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[L4_depth]), name='biases')
        conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weights, strides=[1, 2, 1, 1], padding='SAME') + biases)
    # Pool 2
    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(conv4, ksize=[1,2,1,1], strides=[1,2,1,1], padding='VALID')
    # Fully connected layer
    with tf.name_scope('fc'):
        weights = tf.Variable(tf.truncated_normal([fc_input_size * L4_depth, fc_size], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[fc_size]), name='biases')
        pool2_flat = tf.reshape(pool2, [-1, fc_input_size * L4_depth])
        fc = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)
    # Dropout layer
    with tf.name_scope('dropout'):
        #keep_prob = tf.placeholder(tf.float32)
        fc_drop = tf.nn.dropout(fc, keep_prob)
    # Readout Layer
    with tf.name_scope('readout'):
        weights =  tf.Variable(tf.truncated_normal([fc_size, 2], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[2]), name='biases')
        logits = tf.matmul(fc_drop, weights) + biases
        
    # save inference graph to file
    tf.train.write_graph(tf.get_default_graph().as_graph_def(),TRAIN_DIR, "inference.pbtxt", as_text=True)
        
    return logits
    
def training_graph(logits, labels, learning_rate):
    """Build the training graph.
    
    Args:
        logits: Logits tensor, float - [BATCH_SIZE, NUM_CLASSES].
        labels: Labels tensor, int32 - [BATCH_SIZE, NUM_CLASSES].
        learning_rate: The learning rate for selected optimizer
    Returns:
        train_op: The Op for training.
        loss: The Op for calculating loss.
    """
    # Create an operation that calculates loss.
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels, name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    # Create ADAM optimizer with given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss and increment the global step counter
    train_op = optimizer.minimize(loss, global_step=global_step)
    # save train graph to file
    tf.train.write_graph(tf.get_default_graph().as_graph_def(),TRAIN_DIR, "train.pbtxt", as_text=True)

    
    return train_op, loss
    
FLAGS = None

def main(_):
    # import data
    start_time = time.time()
    
    # calculate input data size
    pooling_size = 2
    input_data_size = int(math.ceil(150 / (math.pow(pooling_size,2))) * (math.pow(pooling_size,2)))
    
    data = data_tools.nlos_classification_dataset(r'\dataset', split_factor = 0.6, scaling = False)
    print("rows: %d, columns: %d" % (data.train.samples.shape[0], data.train.samples.shape[1]))
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # prepare folder to output model
    try:
        os.makedirs(TRAIN_DIR)
    except OSError:
        if os.path.exists(TRAIN_DIR):
            # we are nearly safe
            pass
        else:
            # there was an error on creation
            raise
    
    checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint')
       
    # Build the complete graph for feeding inputs, training, and saving checkpoints
    graph = tf.Graph()
    with graph.as_default():
        # Generate placeholders for input samples and labels.
        x = tf.placeholder(tf.float32, [None, data.train.samples.shape[1]], name='input_data')
        labels = tf.placeholder(tf.float32, [None, 2], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        tf.add_to_collection("inputs", x)       # Remember this Op.
        tf.add_to_collection("labels", labels)  # Remember this Op.
        tf.add_to_collection("keep_prob", keep_prob) # Remember this Op.
        
        # Build a Graph that computes predictions from the inference model.
        logits = inference_graph(x, data.train.samples.shape[1], keep_prob)
        
        # save inference graph to file
        tf.train.write_graph(tf.get_default_graph().as_graph_def(),TRAIN_DIR, "inference.pbtxt", as_text=True)
        
        tf.add_to_collection("logits", logits)  # Remember this Op.
        
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op, loss = training_graph(logits, labels, 1e-4)
        
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
            acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            confusion = tf.contrib.metrics.confusion_matrix(tf.argmax(logits,1), tf.argmax(labels,1), num_classes=2)
        
        # Add the variable initializer Op.
        init = tf.global_variables_initializer()
        
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        
        # save complete flow graph to file
        tf.train.write_graph(tf.get_default_graph().as_graph_def(),TRAIN_DIR, "complete.pbtxt", as_text=True)

    start_time = time.time()

    # Run training for MAX_STEPS and save checkpoint at the end.
    with tf.Session(graph = graph) as sess:
        # Run the Op to initialize the variables.
        sess.run(init)
        
        # 添加损失和准确率的 summary 操作
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', acc_op)

        # 将所有 summary 合并为一个操作
        merged_summary_op = tf.summary.merge_all()

        # 创建一个 summary writer 来将日志写入特定目录
        summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph=sess.graph)

        
        metrics = []
        for step in range(MAX_STEPS):
            # Read a batch of samples and labels.
            samples_batch, labels_batch = data.train.next_batch(BATCH_SIZE)
            # Run one step of the model.
            _, loss_value = sess.run([train_op, loss], feed_dict={x: samples_batch, labels: labels_batch, keep_prob: 0.5})

            # Print loss value.
            if step % 1000 == 0:
                acc_train, conf_mtx_train = sess.run([acc_op, confusion], feed_dict={x: data.train.samples, labels: data.train.labels, keep_prob: 1.0})
                acc_test, conf_mtx_test = sess.run([acc_op, confusion], feed_dict={x: data.test.samples, labels: data.test.labels, keep_prob: 1.0})
                # 计算测试集的损失
                test_loss = sess.run(loss, feed_dict={x: data.test.samples, labels: data.test.labels, keep_prob: 1.0})

                # 计算训练集的 precision、recall 和 F1 值
                true_positive_train = conf_mtx_train[0][0]
                true_negative_train = conf_mtx_train[1][1]
                false_positive_train = conf_mtx_train[0][1]
                false_negative_train = conf_mtx_train[1][0]

                precision_train = float(true_positive_train) / (true_positive_train + false_positive_train + 1e-8)
                recall_train = float(true_positive_train) / (true_positive_train + false_negative_train + 1e-8)
                f1_train = (2 * precision_train * recall_train) / (precision_train + recall_train + 1e-8)

                # 计算测试集的 precision、recall 和 F1 值
                true_positive_test = conf_mtx_test[0][0]
                true_negative_test = conf_mtx_test[1][1]
                false_positive_test = conf_mtx_test[0][1]
                false_negative_test = conf_mtx_test[1][0]

                precision_test = float(true_positive_test) / (true_positive_test + false_positive_test + 1e-8)
                recall_test = float(true_positive_test) / (true_positive_test + false_negative_test + 1e-8)
                f1_test = (2 * precision_test * recall_test) / (precision_test + recall_test + 1e-8)

                # 打印训练集和测试集的性能指标
                print('Step %d: loss = %.2f' % (step, loss_value))
                print("##### epoch: %d, %2f s, train accuracy: %f, test accuracy: %f" % (data.train.epochs_completed, (time.time() - start_time), acc_train, acc_test))
                print("Train Precision: %f, Train Recall: %f, Train F1 Score: %f" % (precision_train, recall_train, f1_train))
                print("Test Precision: %f, Test Recall: %f, Test F1 Score: %f" % (precision_test, recall_test, f1_test))
                print("Train True Positive: %d, Train True Negative: %d, Train False Positive: %d, Train False Negative: %d" % (true_positive_train, true_negative_train, false_positive_train, false_negative_train))
                print("Test True Positive: %d, Test True Negative: %d, Test False Positive: %d, Test False Negative: %d" % (true_positive_test, true_negative_test, false_positive_test, false_negative_test))

                epoch = data.train.epochs_completed  # 记录已经完成的 epoch

                # 创建包含指标的字典
                metric = {
                    '步骤': step,
                    '训练完成轮数': epoch,
                    '训练损失': loss_value,
                    '测试损失': test_loss,
                    '训练准确率': acc_train,
                    '测试准确率': acc_test,
                    '训练真正例': true_positive_train,
                    '训练真负例': true_negative_train,
                    '训练假正例': false_positive_train,
                    '训练假负例': false_negative_train,
                    '测试真正例': true_positive_test,
                    '测试真负例': true_negative_test,
                    '测试假正例': false_positive_test,
                    '测试假负例': false_negative_test,
                    '训练Precision': precision_train,
                    '训练Recall': recall_train,
                    '训练F1值': f1_train,
                    '测试Precision': precision_test,
                    '测试Recall': recall_test,
                    '测试F1值': f1_test
                }
                metrics.append(metric)
                # Write a checkpoint.
                saver.save(sess, checkpoint_file, global_step=step)

            if step % 1000 == 0:
                summary_str = sess.run(
                    merged_summary_op, feed_dict={x: data.test.samples, labels: data.test.labels, keep_prob: 1.0}
                )
                summary_writer.add_summary(summary_str, step)

        # 将metrics列表传递给write_metrics_to_csv函数，并指定CSV文件路径
        write_metrics_to_csv('./origin_metrics.csv', metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data', help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()
