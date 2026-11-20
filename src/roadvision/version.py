"""Version constants that participate in the reproducibility contract."""

__version__ = "0.1.0"

# Bump this when rendering or scene planning changes in a way that can alter
# pixels or ground-truth geometry. The value is stored in every annotation.
GENERATOR_VERSION = "0.1.0"

# Bump this only when the annotation schema changes incompatibly.
SCHEMA_VERSION = "1.0"
