//! File containing struct definitions for a memory representation of MANIFolds
// Will do weird ass things if

use nalgebra::{Const, OVector};
use std::collections::HashMap;

/// Generic struct for N-D points.
pub type Point<const N: usize> = OVector<f64, Const<N>>;

/// Triangle
pub type Triangle = (usize, usize, usize);

#[derive(Clone, Debug, PartialEq)]
/// Abstract representation of a manifold.
pub struct Manifold<const N: usize> {
    vertices: HashMap<usize, Point<N>>,
    triangles: Vec<Triangle>,
}

impl<const N: usize> Manifold<N> {}
