import tensorflow as tf
from tensorflow import keras

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers

from keras import backend
from keras.utils import conv_utils
from keras.utils import tf_utils
from keras.utils import control_flow_util
from keras.engine.input_spec import InputSpec

from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import \
    default_n_bit_quantizers


class QuantConv2DBatchLayer(keras.layers.Layer):

    def __init__(self, conv_layer: keras.layers.Conv2D, bn_layer: keras.layers.BatchNormalization, quantize=True,
                 **kwargs):
        super().__init__()
        self._conv_layer = conv_layer
        self._bn_layer = bn_layer

        # TODO: This copy only structure, not weights
        # TODO: Copy weights
        # Convolution params
        # TODO: I currently do not support more that 1 groups
        self.rank = self._conv_layer.rank
        self.filters = self._conv_layer.filters
        self.kernel_size = self._conv_layer.kernel_size
        self.groups = self._conv_layer.groups
        self.strides = self._conv_layer.strides
        self.padding = self._conv_layer.padding
        self.data_format = self._conv_layer.data_format
        self.dilation_rate = self._conv_layer.dilation_rate
        self.use_bias = self._conv_layer.use_bias

        self.kernel_initializer = self._conv_layer.kernel_initializer
        self.bias_initializer = self._conv_layer.bias_initializer
        self.kernel_regularizer = self._conv_layer.kernel_regularizer
        self.bias_regularizer = self._conv_layer.bias_regularizer
        self.kernel_constraint = self._conv_layer.kernel_constraint
        self.bias_constraint = self._conv_layer.bias_constraint
        self.input_spec = self._conv_layer.input_spec

        self._is_causal = self.padding == "causal"
        self._channels_first = self.data_format == "channels_first"
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2
        )

        # BatchNormalization params
        self.axis = self._bn_layer.axis
        self.momentum = self._bn_layer.momentum
        self.epsilon = self._bn_layer.epsilon
        self.center = self._bn_layer.center
        self.scale = self._bn_layer.scale
        self.beta_initializer = self._bn_layer.beta_initializer
        self.gamma_initializer = self._bn_layer.gamma_initializer
        self.moving_mean_initializer = self._bn_layer.moving_mean_initializer
        self.moving_variance_initializer = self._bn_layer.moving_variance_initializer
        self.beta_regularizer = self._bn_layer.beta_regularizer
        self.gamma_regularizer = self._bn_layer.gamma_regularizer
        self.beta_constraint = self._bn_layer.beta_constraint
        self.gamma_constraint = self._bn_layer.gamma_constraint

        # TODO: this is per channel
        if quantize:
            self.weights_quantizer = default_n_bit_quantizers.DefaultNBitConvWeightsQuantizer()
        else:
            self.weights_quantizer = None

    def build(self, input_shape):
        self.axis = tf_utils.validate_axis(self.axis, input_shape)
        input_shape = tf.TensorShape(input_shape)

        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                "The number of input channels must be evenly divisible by "
                "the number of groups. Received groups={}, but the input "
                "has {} channels (full input shape is {}).".format(
                    self.groups, input_channel, input_shape
                )
            )
        kernel_shape = self.kernel_size + (
            input_channel // self.groups,
            self.filters,
        )

        # self.compute_output_shape contains validations for input_shape
        conv_output_shape = self.compute_output_shape(input_shape)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )

        axis_to_dim = {x: conv_output_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError(
                    "Input has undefined `axis` dimension. Received input "
                    f"with shape {tuple(conv_output_shape)} "
                    f"and axis={tuple(self.axis)}"
                )

        if len(axis_to_dim) == 1:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis
            # dims
            param_shape = [
                axis_to_dim[i] if i in axis_to_dim else 1 for i in range(self.conv_output_shape.rank)
            ]
        self._param_shape = param_shape
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.beta = None

        try:
            # Disable variable partitioning when creating the moving mean and
            # variance
            if hasattr(self, "_scope") and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None
            self.moving_mean = self.add_weight(
                name="moving_mean",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_mean_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False,
            )

            self.moving_variance = self.add_weight(
                name="moving_variance",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_variance_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False,
            )
        finally:
            if partitioner:
                self._scope.set_partitioner(partitioner)

        if self.weights_quantizer is not None:
            self.weights_quantizer.build(input_shape, "weights", self)

        # Copy weights from original convolution and batch norm layer
        conv_weights = self._conv_layer.get_weights()
        conv_weights_len = len(conv_weights)
        bn_weights = self._bn_layer.get_weights()
        current_weights = self.get_weights()
        for i in range(conv_weights_len):
            current_weights[i] = conv_weights[i]

        for j in range(len(conv_weights)):
            current_weights[j + conv_weights_len] = bn_weights[j]

        self.built = True

    def _spatial_output_shape(self, spatial_input_shape):
        return [
            conv_utils.conv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i],
            )
            for i, length in enumerate(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        """
        Copied From: TensorFlow Conv Layer

        Conv+Batch will have same output shape as original convolution layer, so it makes sense to use
        convolution layer function
        """
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        try:
            if self.data_format == "channels_last":
                return tf.TensorShape(
                    input_shape[:batch_rank]
                    + self._spatial_output_shape(input_shape[batch_rank:-1])
                    + [self.filters]
                )
            else:
                return tf.TensorShape(
                    input_shape[:batch_rank]
                    + [self.filters]
                    + self._spatial_output_shape(input_shape[batch_rank + 1:])
                )

        except ValueError:
            raise ValueError(
                "One of the dimensions in the output is <= 0 "
                f"due to downsampling in {self.name}. Consider "
                "increasing the input size. "
                f"Received input shape {input_shape} which would produce "
                "output shape with a zero or negative value in a "
                "dimension."
            )

    def _get_folded_weights(self, variance):
        return (self.gamma / tf.math.sqrt(variance + self.epsilon)) * self.kernel

    def _add_folded_bias(self, outputs, mean, variance, folded=True):
        # TODO: Handle multiple axes batch normalization
        if folded:
            bias = (self.bias - mean) * (
                    self.gamma / tf.math.sqrt(variance + self.epsilon)) + self.beta
        else:
            bias = self.bias
        output_rank = outputs.shape.rank
        if self.rank == 1 and self._channels_first:
            # nn.bias_add does not accept a 1D input tensor.
            bias = tf.reshape(bias, (1, self.filters, 1))
            outputs += bias
        else:
            # Handle multiple batch dimensions.
            if output_rank is not None and output_rank > 2 + self.rank:

                def _apply_fn(o):
                    return tf.nn.bias_add(
                        o, bias, data_format=self._tf_data_format
                    )

                outputs = conv_utils.squeeze_batch_dims(
                    outputs, _apply_fn, inner_rank=self.rank + 1
                )
            else:
                outputs = tf.nn.bias_add(
                    outputs, bias, data_format=self._tf_data_format
                )
        return outputs

    def call(self, inputs, training=None, **kwargs):
        input_shape = inputs.shape

        if training is None:
            training = tf.keras.backend.learning_phase()

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))

        if not training:
            folded_weights = self._get_folded_weights(variance=self.moving_variance)

            # quantization of weights
            if self.weights_quantizer is not None:
                folded_weights = self.weights_quantizer.__call__(inputs, training, folded_weights)

            outputs = self.convolution_op(inputs, folded_weights)
            if self.use_bias:
                outputs = self._add_folded_bias(outputs, self.moving_mean, self.moving_variance)

            if not tf.executing_eagerly() and input_shape.rank:
                # Infer the static output shape:
                out_shape = self.compute_output_shape(input_shape)
                outputs.set_shape(out_shape)

            return outputs

        conv_out = self.convolution_op(inputs, self.kernel)
        if self.use_bias:
            conv_out = self._add_folded_bias(conv_out, 0, 0, False)
        bn_input_shape = conv_out.shape
        ndims = len(bn_input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        batch_mean, batch_variance = tf.nn.moments(conv_out, reduction_axes, keepdims=len(self.axis) > 1)

        # Update moving mean and variance
        new_mean, new_variance = batch_mean, batch_variance

        def _do_update(var, value):
            """Compute the updates for mean and variance."""
            return self._assign_moving_average(
                var, value, self.momentum, input_shape[0]
            )

        def mean_update():
            true_branch = lambda: _do_update(self.moving_mean, new_mean)
            false_branch = lambda: self.moving_mean
            return control_flow_util.smart_cond(
                training, true_branch, false_branch
            )

        def variance_update():
            """Update the moving variance."""
            true_branch = lambda: _do_update(
                self.moving_variance, new_variance
            )

            false_branch = lambda: self.moving_variance
            return control_flow_util.smart_cond(
                training, true_branch, false_branch
            )

        self.add_update(mean_update)
        self.add_update(variance_update)

        folded_weights = self._get_folded_weights(variance=batch_variance)

        # quantization of weights
        if self.weights_quantizer is not None:
            folded_weights = self.weights_quantizer.__call__(inputs, training, folded_weights)

        outputs = self.convolution_op(inputs, folded_weights)

        if self.use_bias:
            outputs = self._add_folded_bias(outputs, batch_mean, batch_variance)

        return outputs

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        def calculate_update_delta():
            decay = tf.convert_to_tensor(1.0 - momentum, name="decay")
            if decay.dtype != variable.dtype.base_dtype:
                decay = tf.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - tf.cast(value, variable.dtype)) * decay
            if inputs_size is not None:
                update_delta = tf.where(
                    inputs_size > 0,
                    update_delta,
                    backend.zeros_like(update_delta),
                )
            return update_delta

        with backend.name_scope("AssignMovingAvg") as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign_sub(calculate_update_delta(), name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign_sub(
                        variable, calculate_update_delta(), name=scope
                    )

    def _assign_new_value(self, variable, value):
        with backend.name_scope("AssignNewValue") as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign(value, name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign(variable, value, name=scope)

    def convolution_op(self, inputs, kernel):
        if self.padding == "causal":
            tf_padding = "VALID"  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding

        return tf.nn.convolution(
            inputs,
            kernel,
            strides=list(self.strides),
            padding=tf_padding,
            dilations=list(self.dilation_rate),
            data_format=self._tf_data_format,
            name=self.__class__.__name__,
        )

    def _compute_causal_padding(self, inputs):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if getattr(inputs.shape, "ndims", None) is None:
            batch_rank = 1
        else:
            batch_rank = len(inputs.shape) - 2
        if self.data_format == "channels_last":
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == "channels_first":
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                f"The input_shape received is {input_shape}, "
                f"where axis {channel_axis} (0-based) "
                "is the channel dimension, which found to be `None`."
            )
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == "causal":
            op_padding = "valid"
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32

    # TODO: get_config(), set_config(), vyřešit přenos váh concat of weights should work
