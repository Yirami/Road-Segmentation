# -*- coding: utf-8 -*-

import tensorflow as tf
import scipy as scp

import vgg_fcn
import utils

log = utils.LOGGER('test_log')
log.add_console()
logger = log.get_logger()

img = scp.misc.imread('./test_data/tabby_cat.png')

with tf.Session() as sess:
    images = tf.placeholder('float')
    feed_dict = {images: img}
    batch_images = tf.expand_dims(images, 0)

    fcn = vgg_fcn.FCN(logger)
    with tf.name_scope("content_vgg"):
        fcn.build(batch_images, debug=True)

    logger.info('Finished building Network.')

    logger.warning("Score weights are initialized random.")
    logger.warning("Do not expect meaningful results.")

    logger.info("Start Initializing Variabels.")

    init = tf.global_variables_initializer()
    sess.run(init)

    logger.info('Running the Network')
    tensors = [fcn.pred, fcn.pred_up]
    down, up = sess.run(tensors, feed_dict=feed_dict)

    down_color = utils.color_image(down[0])
    up_color = utils.color_image(up[0])

    scp.misc.imsave('fcn8_downsampled.png', down_color)
    scp.misc.imsave('fcn8_upsampled.png', up_color)
