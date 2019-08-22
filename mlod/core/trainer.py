"""Detection model trainer.

This file provides a generic training method to train a
DetectionModel.
"""
import datetime
import os
import tensorflow as tf
import time

from mlod.builders import optimizer_builder
from mlod.core import trainer_utils
from mlod.core import summary_utils

slim = tf.contrib.slim


def train(model,
          train_config,
          stagewise_training=False,
          init_checkpoint_dir=None):
    """Training function for detection models.

    Args:
        model: The detection model object.
        train_config: a train_*pb2 protobuf.
        stagewise_training: A Boolean flag indicating stagewise
            training i.e. loading RPN weights onto MLOD model.
        init_checkpoint_dir: Path to the RPN weights to be loaded.
            Required if the `stagewise_training' is enabled.
    """

    model = model
    train_config = train_config
    # Get model configurations
    model_config = model.model_config

    # Create a variable tensor to hold the global step
    global_step_tensor = tf.Variable(
        0, trainable=False, name='global_step')

    #############################
    # Get training configurations
    #############################
    max_iterations = train_config.max_iterations
    summary_interval = train_config.summary_interval
    checkpoint_interval = \
        train_config.checkpoint_interval
    max_checkpoints = train_config.max_checkpoints_to_keep

    paths_config = model_config.paths_config
    logdir = paths_config.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    checkpoint_dir = paths_config.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = checkpoint_dir + '/' + \
        model_config.checkpoint_name

    global_summaries = set([])

    # The model should return a dictionary of predictions
    prediction_dict = model.build() 

    summary_histograms = train_config.summary_histograms
    summary_img_images = train_config.summary_img_images
    summary_bev_images = train_config.summary_bev_images

    ##############################
    # Setup loss
    ##############################
    losses_dict, total_loss = model.loss(prediction_dict)

    # Optimizer
    training_optimizer = optimizer_builder.build(
        train_config.optimizer,
        global_summaries,
        global_step_tensor)

    all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #for var in all_var:
    #    print(var.name)

    sub_bev_var_names = ['bev_vgg_pyr/', 'bev_bottleneck/', 'anchor_predictor/', 
                        'box_classifier_regressor/br0/']

    sub_bev_var_list = []
    for var in sub_bev_var_names:
        sub_bev_var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var)


    sub_img_var_names = ['vgg_16', 'img_bottleneck', 
                        'box_classifier/br1/', 'box_regressor/br1/']

    sub_img_var_list = []
    for var in sub_img_var_names:
        sub_img_var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var)

    fusion_only_var_names = ['box_classifier_regressor/fc6/', 'box_classifier_regressor/fc7/', 'box_classifier_regressor/fc8/','box_classifier_regressor/cls_out/', 
                             'box_classifier_regressor/off_out/', 'box_classifier_regressor/ang_out/']
    fusion_only_var_list = []
    for var in fusion_only_var_names:
        fusion_only_var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var)

    for var in fusion_only_var_list:
        print(var.name)

    # bev, img, fusion 
    training_selection = 'all'

    # Create the train op
    with tf.variable_scope('train_op'):
        if training_selection == 'bev':
            train_op = slim.learning.create_train_op(
                total_loss,
                training_optimizer,
                variables_to_train= sub_bev_var_list,
                clip_gradient_norm=1.0,
                global_step=global_step_tensor)
        elif training_selection == 'img':
            train_op = slim.learning.create_train_op(
                total_loss,
                training_optimizer,
                variables_to_train= sub_img_var_list,
                clip_gradient_norm=1.0,
                global_step=global_step_tensor)
        elif training_selection == 'fusion':
            train_op = slim.learning.create_train_op(
                total_loss,
                training_optimizer,
                variables_to_train=fusion_only_var_list,
                clip_gradient_norm=1.0,
                global_step=global_step_tensor)
        else:
            train_op = slim.learning.create_train_op(
                total_loss,
                training_optimizer,
                clip_gradient_norm=1.0,
                global_step=global_step_tensor)
        
        
        

    # Save checkpoints regularly.
    saver = tf.train.Saver(max_to_keep=max_checkpoints,
                           pad_step_number=True)

    # Add the result of the train_op to the summary
    tf.summary.scalar("training_loss", train_op)

    # Add maximum memory usage summary op
    # This op can only be run on device with gpu
    # so it's skipped on travis
    is_travis = 'TRAVIS' in os.environ
    if not is_travis:
        # tf.summary.scalar('bytes_in_use',
        #                   tf.contrib.memory_stats.BytesInUse())
        tf.summary.scalar('max_bytes',
                          tf.contrib.memory_stats.MaxBytesInUse())

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_merged = summary_utils.summaries_to_keep(
        summaries,
        global_summaries,
        histograms=summary_histograms,
        input_imgs=summary_img_images,
        input_bevs=summary_bev_images
    )

    allow_gpu_mem_growth = train_config.allow_gpu_mem_growth
    if allow_gpu_mem_growth:
        # GPU memory config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = allow_gpu_mem_growth
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()

    # Create unique folder name using datetime for summary writer
    datetime_str = str(datetime.datetime.now())
    logdir = logdir + '/train'
    train_writer = tf.summary.FileWriter(logdir + '/' + datetime_str,
                                         sess.graph)

    if stagewise_training:
        # Load the RPN weights into the model
        # create a separate saver variable to avoid mixing the
        # initial weights to load and current weights to store
        rpn_saver = tf.train.Saver()
        trainer_utils.load_checkpoints(init_checkpoint_dir,
                                       rpn_saver)
        checkpoint_to_restore = rpn_saver.last_checkpoints[-1]
        trainer_utils.load_model_weights(model,
                                         sess,
                                         checkpoint_to_restore)

    # Create init op
    init = tf.global_variables_initializer()

    # Continue from last saved checkpoint
    if not train_config.overwrite_checkpoints:
        trainer_utils.load_checkpoints(checkpoint_dir,
                                       saver)
        if len(saver.last_checkpoints) > 0:
            checkpoint_to_restore = saver.last_checkpoints[-1]
            saver.restore(sess, checkpoint_to_restore)
        else:
            # Initialize the variables
            sess.run(init)
            if train_config.load_vgg_weights:
                #load pre-trained vgg16 weights
                trainer_utils.initialize_vgg(train_config,sess)
    else:
        # Initialize the variables
        sess.run(init)

    # Read the global step if restored
    global_step = tf.train.global_step(sess,
                                       global_step_tensor)
    print('Starting from step {} / {}'.format(
        global_step, max_iterations))

    # Main Training Loop
    last_time = time.time()
    for step in range(global_step, max_iterations + 1):

        # Save checkpoint
        if step % checkpoint_interval == 0:
            global_step = tf.train.global_step(sess,
                                               global_step_tensor)

            saver.save(sess,
                       save_path=checkpoint_path,
                       global_step=global_step)

            print('Step {} / {}, Checkpoint saved to {}-{:08d}'.format(
                step, max_iterations,
                checkpoint_path, global_step))

        # Create feed_dict for inferencing
        feed_dict = model.create_feed_dict()

        # Write summaries and train op
        if step % summary_interval == 0:
            current_time = time.time()
            time_elapsed = current_time - last_time
            last_time = current_time

            if training_selection == 'bev':
                train_op_loss, summary_out = sess.run(
                    [train_op, summary_merged], feed_dict=feed_dict)
            elif training_selection == 'img':
                train_op_loss, summary_out = sess.run(
                    [train_op, summary_merged], feed_dict=feed_dict)
            elif training_selection == 'fusion':
                train_op_loss, summary_out = sess.run(
                    [train_op, summary_merged], feed_dict=feed_dict)
            else:
                train_op_loss, summary_out = sess.run(
                    [train_op, summary_merged], feed_dict=feed_dict)

            print('Step {}, Total Loss {:0.3f}, Time Elapsed {:0.3f} s'.format(
                step, train_op_loss, time_elapsed))
            train_writer.add_summary(summary_out, step)

        else:
            # Run the train op only
            if training_selection == 'bev':
                sess.run(train_op, feed_dict)
            elif training_selection == 'img':
                sess.run(train_op, feed_dict)
            elif training_selection == 'fusion':
                sess.run(train_op, feed_dict)
            else: 
                sess.run(train_op, feed_dict)

    # Close the summary writers
    train_writer.close()
