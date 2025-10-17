"""Package containing task implementations for various robotic environments."""

import os
import toml

from isaaclab_tasks.utils import import_packages


##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

# ensure GR1T2 envs are registered on package import
from .manipulation.pick_place.config import gr1t2  # noqa: F401
from .manipulation.hand_to_hand.hand_to_hand_env_cfg import h2h1
