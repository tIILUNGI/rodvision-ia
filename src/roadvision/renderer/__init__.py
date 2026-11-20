"""Rendering package.

Only lightweight geometry is imported eagerly. Composition imports the generator
package, so importing it from this ``__init__`` would create a cycle.
"""

from .geometry import RoadGeometry

__all__ = ["RoadGeometry"]
