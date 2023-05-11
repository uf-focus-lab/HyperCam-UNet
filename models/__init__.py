# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
from .Conv import Conv
from .U_Net import U_Net
from lib.Module import Module


MODELS: dict[Module] = {
    "Conv": Conv,
    "U_Net": U_Net,
}
