from resnet import *
import tensorflow as tf
from tools.tools import calculate_accuracy
from tools.Tools import calculate_accuracy as eval_accuracy
from classification.Net.ResNet.net_config import Net_Config as net_config
from glob import glob
import shutil

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('load_model_path', '/home/give/PycharmProjects/StomachCanner/classification/Net/ResNet/models/patch_single_connected',
                           '''the model reload path''')
tf.app.flags.DEFINE_string('save_model_path', './models', 'the saving path of the model')
tf.app.flags.DEFINE_string('log_dir', './log/train',
                           """The Summury output directory""")
tf.app.flags.DEFINE_string('log_val_dir', './log/val',
                           """The Summury output directory""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', net_config.BATCH_SIZE, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 10000, "max steps")
tf.app.flags.DEFINE_boolean('resume', True,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


def top_k_error(predictions, labels, k):
    batch_size = float(net_config.BATCH_SIZE) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size


def val(image_tenor, logits, image_values, labels, label_values, save_model_path=None):
    is_training = tf.placeholder('bool', [], name='is_training')
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    loss_ = loss(logits, labels)
    predictions = tf.nn.softmax(logits)
    top1_error = top_k_error(predictions, labels, 1)
    labels_onehot = tf.one_hot(labels, 2)
    accuracy_tensor = calculate_accuracy(predictions, labels_onehot)

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)


    opt = tf.train.MomentumOptimizer(0.01, MOMENTUM)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not True:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.summary.merge_all()

    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    if FLAGS.resume:
        latest = tf.train.latest_checkpoint('/home/give/PycharmProjects/StomachCanner/classification/Net/ResNetHeatMap/models/method5-512')
        if not latest:
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)

    for x in xrange(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)
        i = [loss_]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)
        o = sess.run(i, {
            image_tenor: image_values,
            labels: np.asarray(label_values, np.int32)
        })

        loss_value = o[0]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        top1_error_value, accuracy_value, predictions_values = sess.run(
            [top1_error, accuracy_tensor, predictions],
            feed_dict={
               image_tenor: image_values,
               labels: np.asarray(label_values, np.int32)
            })
        predictions_values = np.argmax(predictions_values, axis=1)
        examples_per_sec = FLAGS.batch_size / float(duration)
        format_str = ('step %d, loss = %.2f, top1 error = %g, accuracy value = %g, 0 rate = %g, 1 rate = %g  (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print predictions_values
        # accuracy = eval_accuracy(predictions_values, label_values)
        # print(format_str % (step, loss_value, top1_error_value, accuracy_value, accuracy[0], accuracy[1], examples_per_sec, duration))