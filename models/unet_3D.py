import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from .attention_layer import AttentionLayer
from .unet3D_layers import (
    UnetDeepSupervision,
    UnetDownsamplingConv,
    UnetGridGatingSignal,
    UnetUpsamplingConv,
)
from .utils import dice_loss, dice_score


class Unet3D(models.Model):
    def __init__(self, use_attention, generator_input_shape, **kwargs):
        super().__init__(**kwargs)
        filters = [16, 32, 64, 128, 256]

        # general config
        self.generator_input_shape = generator_input_shape
        self.use_attention = use_attention

        # downsampling
        self.conv1 = UnetDownsamplingConv(filters[0], True)
        self.conv2 = UnetDownsamplingConv(filters[1], True)
        self.conv3 = UnetDownsamplingConv(filters[2], True)
        self.conv4 = UnetDownsamplingConv(filters[3], True)
        self.center = UnetDownsamplingConv(filters[4], False)
        self.gating = UnetGridGatingSignal(filters[4], (1, 1, 1))

        # attention blocks
        self.attention_block1 = AttentionLayer(filters[1], filters[1], (2, 2, 2))
        self.attention_block2 = AttentionLayer(filters[2], filters[1], (2, 2, 2))
        self.attention_block3 = AttentionLayer(filters[3], filters[1], (2, 2, 2))

        # upsampling
        self.up4 = UnetUpsamplingConv(filters[3])
        self.up3 = UnetUpsamplingConv(filters[2])
        self.up2 = UnetUpsamplingConv(filters[1])
        self.up1 = UnetUpsamplingConv(filters[0])

        # deep supervision
        self.dsv4 = UnetDeepSupervision(1, 8)
        self.dsv3 = UnetDeepSupervision(1, 4)
        self.dsv2 = UnetDeepSupervision(1, 2)
        self.dsv1 = layers.Conv3D(1, 1)

        # final segmentation layer
        self.final = layers.Conv3D(1, 1, activation="sigmoid")

    def call(self, x):
        x = tf.ensure_shape(x, self.generator_input_shape)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        center = self.center(conv4)

        if self.use_attention:
            gating = self.gating(center)
            g_conv4 = self.attention_block3(conv4, gating)
            up4 = self.up4(g_conv4, center)
            g_conv3 = self.attention_block2(conv3, up4)
            up3 = self.up3(g_conv3, up4)
            g_conv2 = self.attention_block1(conv2, up3)
            up2 = self.up2(g_conv2, up3)
            up1 = self.up1(conv1, up2)
        else:
            up4 = self.up4(conv4, center)
            up3 = self.up3(conv3, up4)
            up2 = self.up2(conv2, up3)
            up1 = self.up1(conv1, up2)

        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(tf.concat([dsv1, dsv2, dsv3, dsv4], axis=4))
        return final


class Unet3DFactory:
    def __init__(
        self, input_shape, learning_rate, use_dice_loss, use_attention, **kwargs
    ):
        self.input_shape = (None,) + tuple(input_shape) + (1,)
        self.learning_rate = learning_rate
        self.use_dice_loss = use_dice_loss
        self.use_attention = use_attention

    def produce_unet(self):
        unet_3D = Unet3D(self.use_attention, self.input_shape)
        unet_3D.compile(
            optimizer=optimizers.Adam(self.learning_rate),
            loss=dice_loss if self.use_dice_loss else "binary_crossentropy",
            metrics=[dice_score],
        )
        return unet_3D
