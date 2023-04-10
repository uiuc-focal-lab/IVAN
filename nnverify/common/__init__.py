import re
from enum import Enum

RESULT_DIR = 'results/'


def strip_name(obj, pos=-1):
    return re.split('\.|/', str(obj))[pos]


# Domains used for verification
class Domain(Enum):
    BASE = 1  # This domain is used to represent natural training
    DEEPZ = 2
    DEEPPOLY = 3
    BOX = 4
    LP = 5
    LIRPA_IBP = 6
    LIRPA_CROWN = 7
    LIRPA_CROWN_IBP = 8
    LIRPA_CROWN_OPT = 9


# Used for status of the complete verifier
class Status(Enum):
    VERIFIED = 1
    ADV_EXAMPLE = 2
    UNKNOWN = 3
    MISS_CLASSIFIED = 4
