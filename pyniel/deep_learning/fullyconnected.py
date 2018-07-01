import numpy as np
import tensorflow as tf

from .modelparams import ModelParams
from .custom_ops import n_dimensional_weightmul


class FCNParams(ModelParams):
    def __init__(self):
        self.INPUT_SHAPE = [16]
        self.OUTPUT_SHAPE = [3]
        self.FC_LAYERS = [{"shape": [128]}, {"shape": [128]}, {"shape": [128]}]
        self.LEARNING_RATE = 0.001
        self.DROPOUT = 0.90  # Keep-prob
        self.FLOAT_TYPE = tf.float32
        self.DISABLE_SUMMARY = False


TINY = 1e-8


class FCNetwork(object):
    def __init__(self, model_params):
        self.MP = model_params

        tf.reset_default_graph()
        preset_batch_size = None
        self.variables = []
        self.zero = tf.constant(0)
        self.global_step_tensor = tf.Variable(1, trainable=False, name="global_step")
        self.variables.append(self.global_step_tensor)
        # Graph input
        with tf.name_scope("Placeholders") as scope:
            self.input_placeholder = tf.placeholder(
                self.MP.FLOAT_TYPE,
                shape=[preset_batch_size] + self.MP.INPUT_SHAPE,
                name="input",
            )
            self.ground_truth_placeholder = tf.placeholder(
                self.MP.FLOAT_TYPE,
                shape=[preset_batch_size] + self.MP.OUTPUT_SHAPE,
                name="ground_truth",
            )
            if self.MP.DROPOUT is not None:
                default_dropout = tf.constant(1, dtype=self.MP.FLOAT_TYPE)
                self.dropout_placeholder = tf.placeholder_with_default(
                    default_dropout, (), name="dropout_prob"
                )
        # Input Layer
        previous_layer = self.input_placeholder
        previous_layer_shape = (
            self.MP.INPUT_SHAPE
        )  # Excludes batch dim (which should be at pos 0)
        # Fully connected Layers
        for i, LAYER in enumerate(self.MP.FC_LAYERS):
            previous_layer, previous_layer_shape = self.build_FC_layer(
                LAYER,
                previous_layer,
                previous_layer_shape,
                "Layer" + str(i),
                "Layer" + str(i) + "Weights",
                "_layer_" + str(i),
                activation=tf.nn.relu,
            )

        self.output, _ = self.build_FC_layer(
            {"shape": self.MP.OUTPUT_SHAPE},
            previous_layer,
            previous_layer_shape,
            "Output",
            "OutputWeights",
            "_output_layer",
            activation=tf.nn.relu,
        )

        # Loss
        with tf.name_scope("Loss") as scope:
            # Cross entropy loss of output probabilities vs. input certainties.
            self.loss = tf.losses.softmax_cross_entropy(
                self.ground_truth_placeholder, self.output
            )
            # Average sum of costs over batch.
            self.cost = tf.reduce_mean(self.loss)
            if not self.MP.DISABLE_SUMMARY:
                tf.summary.scalar("cost", self.cost)

        # Optimizers (ADAM)
        with tf.name_scope("Optimizer") as scope:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE)
            self.train_op = self.optimizer.minimize(
                self.cost, global_step=self.global_step_tensor
            )
        # Initialize session
        self.catch_nans = tf.add_check_numerics_ops()
        self.sess = tf.Session()
        self.merged = (
            tf.summary.merge_all() if not self.MP.DISABLE_SUMMARY else self.zero
        )
        tf.global_variables_initializer().run(session=self.sess)
        # Saver
        self.saver = tf.train.Saver(self.variables)
        tf.get_default_graph().finalize()

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        if not self.MP.DISABLE_SUMMARY:
            with tf.name_scope("summaries"):
                mean = tf.reduce_mean(var)
                tf.summary.scalar("mean", mean)
                with tf.name_scope("stddev"):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar("stddev", stddev)
                tf.summary.scalar("max", tf.reduce_max(var))
                tf.summary.scalar("min", tf.reduce_min(var))
                tf.summary.histogram("histogram", var)

    def build_FC_layer(
        self,
        LAYER,
        previous_layer,
        previous_layer_shape,
        scope_name,
        varscope_name,
        var_suffix="",
        reuse=False,
        activation=tf.nn.softplus,
        put_variables_in_list=None,
    ):
        with tf.name_scope(scope_name) as scope:
            layer_shape = LAYER["shape"]
            with tf.variable_scope(varscope_name, reuse=reuse) as varscope:
                weights = tf.get_variable(
                    "weights" + var_suffix,
                    dtype=self.MP.FLOAT_TYPE,
                    shape=previous_layer_shape + layer_shape,
                    initializer=tf.contrib.layers.xavier_initializer(),
                )
                biases = tf.get_variable(
                    "biases" + var_suffix,
                    dtype=self.MP.FLOAT_TYPE,
                    shape=layer_shape,
                    initializer=tf.constant_initializer(0),
                )
            if not reuse:
                self.variables.append(weights)
                self.variables.append(biases)
                if put_variables_in_list is not None:
                    put_variables_in_list.append(weights)
                    put_variables_in_list.append(biases)
                self.variable_summaries(weights)
            layer_output = tf.add(
                n_dimensional_weightmul(
                    previous_layer, weights, previous_layer_shape, layer_shape
                ),
                biases,
            )
            layer_output = activation(layer_output)
            if self.MP.DROPOUT is not None:
                layer_output = tf.nn.dropout(layer_output, self.dropout_placeholder)
        return layer_output, layer_shape

    ## Example functions for different ways to call the model graph.
    def classify_batch(self, batch_input):
        return self.sess.run(
            self.output, feed_dict={self.input_placeholder: batch_input}
        )

    def train_on_single_batch(
        self,
        batch_input,
        batch_ground_truth,
        train_target=None,
        cost_only=False,
        dropout=None,
        summary_writer=None,
    ):
        # feed placeholders
        dict_ = {
            self.input_placeholder: batch_input,
            self.ground_truth_placeholder: batch_ground_truth,
        }
        if self.MP.DROPOUT is not None:
            dict_[self.dropout_placeholder] = (
                self.MP.DROPOUT if dropout is None else dropout
            )
        else:
            if dropout is not None:
                raise ValueError(
                    "This model does not implement dropout yet a value was specified"
                )
        # Graph nodes to target
        cost = [self.cost]
        opt = train_target if train_target is not None else self.train_op
        # compute
        cost, _, _, summary = self.sess.run(
            (cost, opt, self.catch_nans, self.merged), feed_dict=dict_
        )
        if summary_writer is not None:
            summary_writer.add_summary(
                summary, tf.train.global_step(self.sess, self.global_step_tensor)
            )
        return np.array(cost)

    def cost_on_single_batch(
        self, batch_input, batch_ground_truth, summary_writer=None
    ):
        return self.train_on_single_batch(
            batch_input,
            batch_ground_truth,
            train_target=self.zero,
            dropout=1.0,
            summary_writer=summary_writer,
        )
