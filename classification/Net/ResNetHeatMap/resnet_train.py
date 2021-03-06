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
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size


def train(is_training, logits, image_tensor, label_tensor, train_batch, val_batch, save_model_path=None):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    loss_ = loss(logits, label_tensor)
    predictions = tf.nn.softmax(logits)
    print 'predictions shape is ', predictions
    print 'label is ', label_tensor
    top1_error = top_k_error(predictions, label_tensor, 1)
    labels_onehot = tf.one_hot(label_tensor, 2)
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

    tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
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

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(FLAGS.log_val_dir, sess.graph)
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
        i = [train_op, loss_]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)
        train_batch_data = train_batch.next()
        o = sess.run(
            i,
            {
                is_training: True,
                image_tensor: train_batch_data[0],
                label_tensor: train_batch_data[1]
            })

        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            top1_error_value, accuracy_value, predictions_values = sess.run(
                [top1_error, accuracy_tensor, predictions],
                feed_dict={
                    is_training: True,
                    image_tensor: train_batch_data[0],
                    label_tensor: train_batch_data[1]
                })
            predictions_values = np.argmax(predictions_values, axis=1)
            examples_per_sec = FLAGS.batch_size / float(duration)
            accuracy = eval_accuracy(predictions_values, train_batch_data[1])
            format_str = ('step %d, loss = %.2f, top1 error = %g, accuracy value = %g, 0 rate = %g, 1 rate = %g  (%.1f examples/sec; %.3f '
                          'sec/batch)')

            print(format_str % (step, loss_value, top1_error_value, accuracy_value, accuracy[0], accuracy[1], examples_per_sec, duration))
        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 10 == 0:
            checkpoint_path = os.path.join(save_model_path, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)
            save_dir = os.path.join(save_model_path, str(step))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            filenames = glob(os.path.join(save_model_path, '*-'+str(int(step + 1))+'.*'))
            for filename in filenames:
                shutil.copy(
                    filename,
                    os.path.join(save_dir, os.path.basename(filename))
                )
        # Run validation periodically
        if step > 1 and step % 10 == 0:
            val_batch_data = val_batch.next()
            _, top1_error_value, summary_value, accuracy_value, predictions_values = \
                sess.run([val_op, top1_error, summary_op, accuracy_tensor, predictions],
                         {
                             is_training: False,
                             image_tensor: val_batch_data[0],
                             label_tensor: val_batch_data[1]
                         })
            predictions_values = np.argmax(predictions_values, axis=1)
            accuracy = eval_accuracy(predictions_values, val_batch_data[1])
            print('Validation top1 error %.2f, accuracy value %f,  0 rate = %g, 1 rate = %g  '
                  % (top1_error_value, accuracy_value, accuracy[0], accuracy[1]))
            val_summary_writer.add_summary(summary_value, step)