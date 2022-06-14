from tensorflow.keras import layers, backend


class GridAttentionLayer(layers.Layer):
    def __init__(self, input_size, inter_size, sub_sample_factor):
        super().__init__()
        self.sub_sample_factor = sub_sample_factor

        self.output_transform_conv = layers.Conv3D(
            input_size, kernel_size=1, strides=1, padding="valid"
        )
        self.output_transform_batch_norm = layers.BatchNormalization()
        self.theta = layers.Conv3D(
            inter_size,
            kernel_size=sub_sample_factor,
            strides=sub_sample_factor,
            padding="valid",
            use_bias=False,
        )
        self.phi = layers.Conv3D(
            inter_size,
            kernel_size=1,
            strides=1,
            padding="valid",
        )
        self.psi = layers.Conv3D(1, kernel_size=1, strides=1, padding="valid")

    def call(self, input_features, gating_signal):
        transformed_input_features = self.theta(input_features)
        transformed_gating_signal = self.phi(gating_signal)
        channels_indexes = [1, 2, 3]
        gating_signal_down_sample_size = [
            transformed_gating_signal.shape[i] // transformed_input_features.shape[i]
            for i in channels_indexes
        ]
        transformed_gating_signal = layers.MaxPooling3D(
            pool_size=tuple(gating_signal_down_sample_size)
        )(transformed_gating_signal)
        attention = layers.Activation("relu")(
            transformed_input_features + transformed_gating_signal
        )
        attention = layers.Activation("sigmoid")(self.psi(attention))
        attention_up_sample_size = [
            input_features.shape[i] // attention.shape[i] for i in channels_indexes
        ]
        attention = layers.UpSampling3D(size=tuple(attention_up_sample_size))(attention)
        attention = backend.repeat_elements(attention, input_features.shape[4], axis=4)
        attention_padding = [
            (0, input_features.shape[i] % attention.shape[i]) for i in channels_indexes
        ]
        attention = layers.ZeroPadding3D(padding=attention_padding)(attention)
        gate = attention * input_features
        gate = self.output_transform_conv(gate)
        gate = self.output_transform_batch_norm(gate)
        return gate


class AttentionLayer(layers.Layer):
    def __init__(self, input_size, inter_size, sub_sample_factor):
        super().__init__()
        self.grid_attention_block = GridAttentionLayer(
            input_size, inter_size, sub_sample_factor
        )
        self.combine_layers = [
            layers.Conv3D(input_size, kernel_size=1, strides=1, padding="valid"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
        ]

    def call(self, input_features, gating_signal):
        gate = self.grid_attention_block(input_features, gating_signal)
        for combine_layer in self.combine_layers:
            gate = combine_layer(gate)
        return gate
