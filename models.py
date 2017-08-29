import tensorflow as tf

reuse_variables = None

def load_model(ckpt_dir, batch_size, input_node, labels_node=None, first_var=0):
    global reuse_variables
    if ckpt_dir in ['log', 'log_adv', 'log_ensadv']:
        import resnet_model_reusable
        hps = resnet_model_reusable.HParams(
            batch_size=batch_size,
            num_classes=10,
            min_lrn_rate=None,
            lrn_rate=None,
            num_residual_units=5,
            use_bottleneck=False,
            weight_decay_rate=0.,
            relu_leakiness=0.1,
            optimizer=None,
        )
        input_scaled = tf.map_fn(lambda image: tf.image.per_image_standardization(image), input_node)
        m = resnet_model_reusable.ResNet(hps, input_scaled, labels_node, 'eval', reuse_variables=reuse_variables)
        m._build_model()
        my_vars = tf.global_variables()[first_var:]
        if labels_node is not None:
            m._build_cost()
        reuse_variables = True
        class Net(object):
            def get_logits(self):
                return m.logits
            def get_loss(self):
                return m.cost_no_decay # %%%
            def load(self, session):
                saver = tf.train.Saver(my_vars)
                ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
                saver.restore(session, ckpt_state.model_checkpoint_path)
        return Net()
    elif ckpt_dir in ['log_wide', 'log_adv_wide', 'log_ensadv_wide']:
        import resnet_model_reusable_wide
        hps = resnet_model_reusable_wide.HParams(
            batch_size=batch_size,
            num_classes=10,
            min_lrn_rate=None,
            lrn_rate=None,
            num_residual_units=4,
            use_bottleneck=False,
            weight_decay_rate=0.,
            relu_leakiness=0.1,
            optimizer=None,
        )
        input_scaled = tf.map_fn(lambda image: tf.image.per_image_standardization(image), input_node)
        m = resnet_model_reusable_wide.ResNet(hps, input_scaled, labels_node, 'eval', reuse_variables=reuse_variables)
        m._build_model()
        if labels_node is not None:
            m._build_cost()
        my_vars = tf.global_variables()[first_var:]
        reuse_variables = True
        class Net(object):
            def get_logits(self):
                return m.logits
            def get_loss(self):
                return m.cost
            def load(self, session):
                saver = tf.train.Saver(my_vars)
                ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
                saver.restore(session, ckpt_state.model_checkpoint_path)
        return Net()
    elif ckpt_dir in ['tutorial/train', 'tutorial/train_adv', 'tutorial/train_ensadv']:
        import tutorial.cifar10_reusable
        tutorial.cifar10_reusable.FLAGS.batch_size = batch_size
        logits = tutorial.cifar10_reusable.inference(input_node)
        if labels_node is not None:
            labels_sparse = tf.argmax(labels_node, axis=1)
            loss = tutorial.cifar10_reusable.loss(logits, labels_sparse)
        my_vars = tf.global_variables()[first_var:]
        reuse_variables = True
        class Net(object):
            def get_logits(self):
                return logits
            def get_loss(self):
                return loss
            def load(self, session):
                saver = tf.train.Saver(my_vars)
                ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
                saver.restore(session, ckpt_state.model_checkpoint_path)
        return Net()
    else:
        raise
