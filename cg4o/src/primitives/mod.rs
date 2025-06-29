/// A module for geometric primitives and their operations.

use crate::math::{Float, Vec3};

/// A 2D surface.
trait Surface {
    /// Returns the surface sag at a given position.
    fn sag(&self, position: Vec3) -> Float;
}