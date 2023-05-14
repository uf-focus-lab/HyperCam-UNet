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
from .loss import *


MODELS: dict[str, type[Module]] = {
    "Conv": Conv,
    "U_Net": U_Net,
}
