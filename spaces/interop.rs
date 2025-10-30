//! Numeric Backends Interop (Step 10 of README)
//! Optional conversions between BoxSpace payloads and ndarray/nalgebra types.
//!
//! This module is intentionally small and fully gated behind feature flags
//! to avoid hard dependencies. The core crate continues to use plain arrays
//! (e.g., `[T; N]`) for BoxSpace elements.

use crate::spaces::BoxSpace;

// ndarray interop
#[cfg(feature = "ndarray")]
pub mod ndarray_impl {
    use super::*;
    use ndarray::Array1;

    /// Error type for conversions from ndarray to fixed-size arrays.
    #[derive(Debug, Clone)]
    pub struct NdarrayShapeError;

    impl<T: Copy + PartialOrd + Clone, const N: usize> BoxSpace<T, N> {
        /// Convert a BoxSpace element `[T; N]` into an `ndarray::Array1<T>`.
        pub fn to_ndarray(elem: [T; N]) -> Array1<T> {
            Array1::from_vec(elem.to_vec())
        }

        /// Attempt to convert an `ndarray::Array1<T>` into a BoxSpace element `[T; N]`.
        pub fn from_ndarray(arr: &Array1<T>) -> Result<[T; N], NdarrayShapeError> {
            if arr.len() != N { return Err(NdarrayShapeError); }
            let vec = arr.to_vec();
            vec.try_into().map_err(|_| NdarrayShapeError)
        }
    }
}

// nalgebra interop
#[cfg(feature = "nalgebra")]
pub mod nalgebra_impl {
    use super::*;
    use nalgebra::SVector;

    impl<T: nalgebra::Scalar + Copy + PartialOrd, const N: usize> BoxSpace<T, N> {
        /// Convert a BoxSpace element `[T; N]` into an `nalgebra::SVector<T, N>`.
        pub fn to_nalgebra(elem: [T; N]) -> SVector<T, N> {
            SVector::<T, N>::from_row_slice(&elem)
        }

        /// Convert an `nalgebra::SVector<T, N>` into a BoxSpace element `[T; N]`.
        pub fn from_nalgebra(v: &SVector<T, N>) -> [T; N] {
            let mut out: [T; N] = [v[0]; N];
            for i in 0..N { out[i] = v[i]; }
            out
        }
    }
}
