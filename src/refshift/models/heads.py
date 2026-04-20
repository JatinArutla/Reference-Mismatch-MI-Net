from tensorflow.keras.layers import Dense, Activation, Lambda, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf

def build_classifier_head(x, n_classes: int, from_logits: bool = False):
    logits = Dense(n_classes)(x)
    act = "linear" if from_logits else "softmax"
    return Activation(act)(logits)

def build_projection_head(
    ssl_feat,
    proj_dim: int = 256,
    out_dim: int = 128,
    *,
    l2norm: bool = True,
    use_bn: bool = True,
    n_layers: int = 3,
):
    """SSL projection head (expander).

    For VICReg/Barlow: use_bn=True, l2norm=False, n_layers=3
    For SimCLR/NT-Xent: use_bn=True, l2norm=True, n_layers=2

    The VICReg paper uses 3-layer MLP with BatchNorm between layers.
    Without BN, the encoder collapses because variance regularization
    gradients get absorbed by the projector before reaching the backbone.
    """
    h = ssl_feat
    for i in range(n_layers - 1):
        h = Dense(proj_dim, use_bias=False if use_bn else True, name=f"proj_dense_{i}")(h)
        if use_bn:
            h = BatchNormalization(name=f"proj_bn_{i}")(h)
        h = Activation("relu", name=f"proj_relu_{i}")(h)
    # Final layer: no activation, no BN (VICReg paper convention)
    z = Dense(out_dim, activation=None, name="proj_out")(h)
    if l2norm:
        z = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name="l2norm")(z)
    return z