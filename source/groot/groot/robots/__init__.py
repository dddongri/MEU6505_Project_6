"""Package containing asset and sensor configurations."""

import os
import toml

##
# Configuration for different assets.
##

# Conveniences to other module directories via relative paths
GR1T2_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
"""Path to the extension source directory."""