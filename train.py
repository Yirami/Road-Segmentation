# -*- coding: utf-8 -*-
import os
import tensorflow as tf

import vgg_fcn
import batch_generator as dgen
import utils
import evaluate

# import collections

# flags = tf.app.flags
# FLAGS = flags.FLAGS

learning_rate = 1e-4
debug = True
vgg16_weights_path = r'weights/vgg16.npy'
num_classes = 2
data_dir = 'data'
logs_dir = 'trace'
max_iteration = int(5e4+1)

def train(loss, trainable_vars):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss, var_list=trainable_vars)
    if debug:
        print(len(trainable_vars))
        for grad, var in grads:
            vgg_fcn._gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)



if __name__ == '__main__':
    log = utils.LOGGER('train_log')
    log.add_console()
    log.add_file('train_log.txt')
    logger = log.get_logger()
    logger.info('-------------------- New Train Start ! ----------------------')
    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, name='input_images')
        gts = tf.placeholder(tf.int32, name='input_gts')

        fcn_net = vgg_fcn.FCN(logger, vgg16_weights_path = vgg16_weights_path)
        with tf.name_scope('content_net'):
            fcn_net.build(images, logger, train=True, num_classes=num_classes, debug=True)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    logits=fcn_net.upscore32,
                                                    labels=gts,
                                                    name='entropy'))
        tf.summary.scalar('entropy', loss)

        trainable_var = tf.trainable_variables()
        if debug:
            for var in trainable_var:
                vgg_fcn._regularization_summary(var)
        train_op = train(loss, trainable_var)

        summary_op = tf.summary.merge_all()

        logger.info('Initializing Saver ...')
        saver = tf.train.Saver(max_to_keep=None)

        summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('Model restored ...')

        records_list = utils.make_records_list_for_KITTI(data_dir)
        data_generator = dgen.Dataset(records_list, logger, batch_size=2)

        opt = utils.TO_SAVE()
        for itr in range(max_iteration):
            train_imgs, train_gts, train_names = data_generator.next_batch()
            feed_dict = {images: train_imgs, gts: train_gts}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                logger.info('Step: %d, Train_loss: %g' % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 100 == 0:
                val_imgs, val_gts, val_names = data_generator.next_val_batch()
                val_loss, output_imgs = sess.run([loss, fcn_net.pred_with_softmax], feed_dict={images: val_imgs, gts: val_gts})
                logger.info('Step: %d, Validation_loss: %g' % (itr, val_loss))
                logger.debug('Step: %d, output images shape: %s' % (itr, str(output_imgs.shape)))
                if opt.maybe_save(val_loss):
                    saver.save(sess, os.path.join(logs_dir, 'model.ckpt'), itr)
