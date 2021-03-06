# -*- coding: utf-8 -*-

import tensorflow as tf

import utils

def evaluate_routine(input_images, output_images, image_names, postfix, threshold=0.5):
    assert input_images.shape[:3] == output_images.shape[:3]
    for idx in range(input_images.shape[0]):
        image = input_images[idx]
        shape = image.shape
        confidence = output_images[idx,:,:, 1].reshape(shape[0], shape[1])
        prediction = confidence > threshold
        product = utils.overlay_for_2_classes(image, prediction)
        utils.save_images(product, image_names[idx], postfix=postfix)

def restore_and_evaluation(meta_dir, ckpt_dir, batch_size=2):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_dir)
        saver.restore(sess, ckpt_dir)
        graph = tf.get_default_graph()
        images = graph.get_tensor_by_name('input_images:0')
        softmax = graph.get_tensor_by_name('content_net/pred_softmax:0')
        imgs_all, names_all = utils.get_evaluate_images()
        eval_grps = utils.make_groups_for_evaluate(imgs_all, names_all, batch_size)
        for imgs, names in eval_grps:
            feed_dict = {images:imgs}
            out_imgs = sess.run(softmax, feed_dict=feed_dict)
            evaluate_routine(imgs, out_imgs, names, '-eval')

if __name__ == '__main__':
    meta_dir = 'trace/model.ckpt-1500.meta'
    ckpt_dir = 'trace/model.ckpt-1500'
    restore_and_evaluation(meta_dir, ckpt_dir)
