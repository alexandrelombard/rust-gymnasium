// Common traits for Gymnasium spaces (Step 4)

use rand::Rng;

/// A trait implemented by all spaces.
/// Element is the value type that lives in the space (e.g., u32 for Discrete,
/// or [T; N] for a fixed-size BoxSpace).
pub trait Space {
    type Element;

    /// Draw a sample from the space using the provided RNG.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::Element;

    /// Return true if the given element is a valid member of the space.
    fn contains(&self, elem: &Self::Element) -> bool;
}
