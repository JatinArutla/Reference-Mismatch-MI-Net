from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from .heads import build_projection_head


# Per-loss projector defaults matching each paper's original recipe.
#
# SimCLR (NT-Xent): 2-layer MLP, no BatchNorm, L2-normalised output, temperature-scaled.
#   Chen et al. 2020, SimCLR. The projector head is described as 2 layers with ReLU,
#   and the features are L2-normalised before computing cosine similarity.
#
# Barlow Twins: 3-layer MLP with BatchNorm between layers, NO L2-norm on output.
#   Zbontar et al. 2021. The loss operates on the cross-correlation of batch-normalised
#   features; explicit L2-norm would double-normalise.
#
# VICReg: 3-layer MLP with BatchNorm between layers, NO L2-norm on output.
#   Bardes et al. 2022. Variance and covariance regularisation act directly on
#   raw projector outputs; L2-norm would collapse the variance term to a constant.
PROJECTOR_CONFIGS: dict[str, dict] = {
    "ntxent":  {"n_layers": 2, "use_bn": False, "l2norm": True},
    "simclr":  {"n_layers": 2, "use_bn": False, "l2norm": True},
    "nt_xent": {"n_layers": 2, "use_bn": False, "l2norm": True},
    "barlow":  {"n_layers": 3, "use_bn": True,  "l2norm": False},
    "barlow_twins": {"n_layers": 3, "use_bn": True, "l2norm": False},
    "vicreg":  {"n_layers": 3, "use_bn": True,  "l2norm": False},
}


def projector_config_for_loss(ssl_loss: str) -> dict:
    """Return the canonical projector kwargs for ``ssl_loss``.

    Raises KeyError with a helpful message if the loss is unknown.
    """
    key = (ssl_loss or "").lower().strip()
    if key not in PROJECTOR_CONFIGS:
        raise KeyError(
            f"No canonical projector config for ssl_loss={ssl_loss!r}. "
            f"Known: {sorted(PROJECTOR_CONFIGS)}"
        )
    return dict(PROJECTOR_CONFIGS[key])


def build_ssl_projector(
    encoder_with_tap: Model,
    proj_dim: int = 256,
    out_dim: int = 128,
    *,
    l2norm: bool = True,
    use_bn: bool = True,
    n_layers: int = 3,
) -> Model:
    inp = Input(shape=encoder_with_tap.input_shape[1:], name="ssl_in")
    out_pred, ssl_feat = encoder_with_tap(inp)
    z = build_projection_head(
        ssl_feat,
        proj_dim=proj_dim,
        out_dim=out_dim,
        l2norm=l2norm,
        use_bn=use_bn,
        n_layers=n_layers,
    )
    return Model(inp, z, name="SSL_Projector")


def build_ssl_projector_for_loss(
    encoder_with_tap: Model,
    ssl_loss: str,
    *,
    proj_dim: int = 256,
    out_dim: int = 128,
) -> Model:
    """Build an SSL projector with the canonical config for the given loss.

    This is the preferred entry point: it guarantees the projector matches the
    loss's original paper (SimCLR: no-BN + L2; Barlow/VICReg: BN, no L2).
    """
    cfg = projector_config_for_loss(ssl_loss)
    return build_ssl_projector(
        encoder_with_tap,
        proj_dim=proj_dim,
        out_dim=out_dim,
        **cfg,
    )