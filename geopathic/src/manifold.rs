//! File containing struct definitions for a memory representation of MANIFolds
// Will do weird ass things if

use nalgebra::DVector;
use std::collections::HashMap;

/// Generic struct for N-D points.
pub type Point = DVector<f32>;

/// Polygon
pub type Polygon = Vec<usize>;

#[derive(Clone, Debug, PartialEq)]
/// Abstract representation of a manifold.
pub struct Manifold {
    pub(crate) vertices: HashMap<usize, Point>,
    pub(crate) faces: Vec<Polygon>,
}

impl Manifold {
    pub fn new(vertices: HashMap<usize, Point>, faces: Vec<Polygon>) -> Self {
        Manifold { vertices, faces }
    }
}
