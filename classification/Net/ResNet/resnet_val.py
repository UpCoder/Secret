from resnet import *
import tensorflow as tf
from tools.Tools import get_game_evaluate

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('save_model_dir', './models', 'the path using to save model')
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 16, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
tf.app.flags.DEFINE_boolean('resume', True,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size


def val(is_training, logits, images, labels, k=1, is_testing=False):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    predictions = tf.nn.softmax(logits)

    top1_error = top_k_error(predictions, labels, k)

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)


    saver = tf.train.Saver(tf.all_variables())

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    sess.run(tf.initialize_local_variables())
    tf.train.start_queue_runners(sess=sess)


    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.save_model_dir)
        if not latest:
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)
    count = 0
    acc_sum = 0.0
    logits_value = []
    labels_value = []
    while True:
        try:
            count += 1
            _, top1_error_value, batch_logits_value, batch_labels_value = sess.run([val_op, top1_error, predictions, labels], {is_training: False})
            acc_sum += top1_error_value
            logits_value.extend(batch_logits_value)
            labels_value.extend(batch_labels_value)
            if not is_testing:
                print('Validation top%d error %.2f, count is %d' % (k, top1_error_value, count))
        except Exception, _:
            break
    if not is_testing:
        print 'The top%d error on the total validation set: %f' % (k, acc_sum/count)
        print labels_value
        print np.shape(logits_value)
        recall, precesion, f1_score = get_game_evaluate(logits=np.argmax(logits_value, axis=1), labels=labels_value)
        print 'recall: %f, precision: %f, f1_score: %f' % (recall, precesion, f1_score)
    if is_testing:
        prediction_flags = np.argmax(logits_value, axis=1)
        return prediction_flags

