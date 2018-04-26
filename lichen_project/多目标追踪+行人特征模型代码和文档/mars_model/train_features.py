# vim: expandtab:ts=4:sw=4
import os
import errno
import numpy as np
import cv2
import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim


def _batch_norm_fn(x, scope=None):
    if scope is None:
        scope = tf.get_variable_scope().name + "/bn"
    return slim.batch_norm(x, scope=scope)


def create_link(
        incoming, network_builder, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
        regularizer=None, is_first=False, summarize_activations=True):
    if is_first:
        network = incoming
    else:
        network = _batch_norm_fn(incoming, scope=scope + "/bn")
        network = nonlinearity(network)
        if summarize_activations:
            tf.summary.histogram(scope+"/activations", network)

    pre_block_network = network
    post_block_network = network_builder(pre_block_network, scope)

    incoming_dim = pre_block_network.get_shape().as_list()[-1]
    outgoing_dim = post_block_network.get_shape().as_list()[-1]
    if incoming_dim != outgoing_dim:
        assert outgoing_dim == 2 * incoming_dim, \
            "%d != %d" % (outgoing_dim, 2 * incoming)
        projection = slim.conv2d(
            incoming, outgoing_dim, 1, 2, padding="SAME", activation_fn=None,
            scope=scope+"/projection", weights_initializer=weights_initializer,
            biases_initializer=None, weights_regularizer=regularizer)
        network = projection + post_block_network
    else:
        network = incoming + post_block_network
    return network


def create_inner_block(
        incoming, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(1e-3),
        bias_initializer=tf.zeros_initializer(), regularizer=None,
        increase_dim=False, summarize_activations=True):
    n = incoming.get_shape().as_list()[-1]
    stride = 1
    if increase_dim:
        n *= 2
        stride = 2

    incoming = slim.conv2d(
        incoming, n, [3, 3], stride, activation_fn=nonlinearity, padding="SAME",
        normalizer_fn=_batch_norm_fn, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/1")
    if summarize_activations:
        tf.summary.histogram(incoming.name + "/activations", incoming)

    incoming = slim.dropout(incoming, keep_prob=0.6)

    incoming = slim.conv2d(
        incoming, n, [3, 3], 1, activation_fn=None, padding="SAME",
        normalizer_fn=None, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/2")
    return incoming


def residual_block(incoming, scope, nonlinearity=tf.nn.elu,
                   weights_initializer=tf.truncated_normal_initializer(1e3),
                   bias_initializer=tf.zeros_initializer(), regularizer=None,
                   increase_dim=False, is_first=False,
                   summarize_activations=True):

    def network_builder(x, s):
        return create_inner_block(
            x, s, nonlinearity, weights_initializer, bias_initializer,
            regularizer, increase_dim, summarize_activations)

    return create_link(
        incoming, network_builder, scope, nonlinearity, weights_initializer,
        regularizer, is_first, summarize_activations)


def _create_network(incoming, num_classes, reuse=None, l2_normalize=True,
                   create_summaries=True, weight_decay=1e-8):
    nonlinearity = tf.nn.elu
    conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = slim.l2_regularizer(weight_decay)
    fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    fc_bias_init = tf.zeros_initializer()
    fc_regularizer = slim.l2_regularizer(weight_decay)

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    network = incoming
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_1",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)
        tf.summary.image("conv1_1/weights", tf.transpose(
            slim.get_variables("conv1_1/weights:0")[0], [3, 0, 1, 2]),
                         max_outputs=128)
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_2",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)

    network = slim.max_pool2d(network, [3, 3], [2, 2], scope="pool1")

    network = residual_block(
        network, "conv2_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False, is_first=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv2_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv3_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv3_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv4_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv4_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    feature_dim = network.get_shape().as_list()[-1]
    print("feature dimensionality: ", feature_dim)
    network = slim.flatten(network)

    network = slim.dropout(network, keep_prob=0.6)
    network = slim.fully_connected(
        network, feature_dim, activation_fn=nonlinearity,
        normalizer_fn=batch_norm_fn, weights_regularizer=fc_regularizer,
        scope="fc1", weights_initializer=fc_weight_init,
        biases_initializer=fc_bias_init)

    features = network

    if l2_normalize:
        # Features in rows, normalize axis 1.
        features = slim.batch_norm(features, scope="ball", reuse=reuse)
        feature_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(features), [1], keep_dims=True))
        features = features / feature_norm

        with slim.variable_scope.variable_scope("ball", reuse=reuse):
            weights = slim.model_variable(
                "mean_vectors", (feature_dim, num_classes),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale", (num_classes, ), tf.float32,
                tf.constant_initializer(0., tf.float32), regularizer=None)
            if create_summaries:
                tf.summary.histogram("scale", scale)
            #scale = slim.model_variable(
            #    "scale", (), tf.float32,
            #    initializer=tf.constant_initializer(0., tf.float32),
            #    regularizer=slim.l2_regularizer(1e-2))
            #if create_summaries:
            #    tf.scalar_summary("scale", scale)
            scale = tf.nn.softplus(scale)

        # Each mean vector in columns, normalize axis 0.
        weight_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(weights), [0], keep_dims=True))
        logits = scale * tf.matmul(features, weights / weight_norm)

    else:
        logits = slim.fully_connected(
            features, num_classes, activation_fn=None,
            normalizer_fn=None, weights_regularizer=fc_regularizer,
            scope="softmax", weights_initializer=fc_weight_init,
            biases_initializer=fc_bias_init)

    return features, logits


def _network_factory(num_classes, is_training, weight_decay=1e-8):

    def factory_fn(image, reuse, l2_normalize):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                     slim.batch_norm, slim.layer_norm],
                                    reuse=reuse):
                    features, logits = _create_network(
                        image, num_classes, l2_normalize=l2_normalize,
                        reuse=reuse, create_summaries=is_training,
                        weight_decay=weight_decay)
                    return features, logits

    return factory_fn


def _preprocess(image, is_training=False, enable_more_augmentation=True):
    image = image[:, :, ::-1]  # BGR to RGB
    if is_training:
        image = tf.image.random_flip_left_right(image)
        if enable_more_augmentation:
            image = tf.image.random_brightness(image, max_delta=50)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


def create_image_trainer(image_shape, num_classes, epochs, batch_size=32,
                         learning_rate_base=0.01, learning_rate_decay_interval=7500,
                         learning_rate_decay=0.99, checkpoint_path=None,
                         model_save_path='model',
                         max_to_keep=100, log_file_path='log/train.log'):
    image_var = tf.placeholder(tf.uint8, (None, ) + image_shape)
    labels_ = tf.placeholder(tf.float32, (None, num_classes))

    preprocessed_image_var = tf.map_fn(
        lambda x: _preprocess(x, is_training=True),
        tf.cast(image_var, tf.float32))

    l2_normalize = True
    factory_fn = _network_factory(
        num_classes=num_classes, is_training=True, weight_decay=1e-8)
    feature_var, logits_var = factory_fn(
        preprocessed_image_var, l2_normalize=l2_normalize, reuse=None)
    feature_dim = feature_var.get_shape().as_list()[-1]
    classification_loss = slim.losses.softmax_cross_entropy(logits_var, labels_)
    total_loss = slim.losses.get_total_loss(add_regularization_losses=True)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        learning_rate_decay_interval,
        learning_rate_decay)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(total_loss, global_step=global_step)

    session = tf.Session()
    #from tensorflow.python import debug as tf_debug
    #session = tf_debug.LocalCLIDebugWrapperSession(session)

    if checkpoint_path is not None:
        saver = tf.train.Saver(slim.get_variables_to_restore()) 
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            checkpoint_path, slim.get_variables_to_restore())
        session.run(init_assign_op, feed_dict=init_feed_dict)
    else:
        session.run(tf.global_variables_initializer())

    def trainer(data_x_paths, data_y):
        data_y = np.asarray(data_y)
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        
        logging.basicConfig(filename=log_file_path, filemode="w", level=logging.DEBUG)
        data_len = len(data_x_paths)
        num_batches = int(data_len / batch_size)

        for epoch in range(epochs):
            indexs = np.arange(data_len)
            np.random.shuffle(indexs)
            s, e = 0, 0
            for i in range(num_batches):
                s, e = i * batch_size, (i + 1) * batch_size
                data_x_batch = np.asarray([cv2.resize(cv2.imread(data_x_paths[index], cv2.IMREAD_COLOR), (image_shape[1], image_shape[0])) for index in indexs[s:e]])
                data_y_batch = data_y[indexs[s:e]]
                batch_data_dict = {image_var: data_x_batch, labels_: data_y_batch}
                _, loss_value, global_step_value = session.run([train_step, total_loss, global_step], feed_dict=batch_data_dict)
                print(global_step_value, loss_value)
                logging.info("%d: %g" % (global_step_value, loss_value))
                if (i+1) % 1000 == 0:
                    print("save model checkpoint for %d..." % global_step_value)
                    saver.save(session, os.path.join(model_save_path, 'model.ckpt'), global_step=global_step)
        
            if e < data_len:
                data_x_batch = np.asarray([cv2.resize(cv2.imread(data_x_paths[index], cv2.IMREAD_COLOR), (image_shape[1], image_shape[0])) for index in indexs[e:]])
                data_y_batch = data_y[indexs[s:e]]
                batch_data_dict = {image_var: data_x_batch, labels_: data_y_batch}
                _, loss_value, global_step_value = session.run([train_step, total_loss, global_step], feed_dict=batch_data_dict)
                print(global_step_value, loss_value)
                logging.info("%d: %g" % (global_step_value, loss_value))

            print("save model checkpoint for %d..." % global_step_value)
            saver.save(session, os.path.join(model_save_path, 'model.ckpt'), global_step=global_step)

    return trainer
