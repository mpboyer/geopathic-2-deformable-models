//! Implementation of the Improved Chen-Han (ICH) algorithm for pathfinding.

use crate::manifold::Manifold;
use nalgebra::Point3;

struct Window {
    edge: usize,
    // first endpoint of the edge
    b0: f64,
    // second endpoint of the edge
    b1: f64,
    // distance from the first endpoint to the pseudosource
    d0: f64,
    // distance from the second endpoint to the pseudosource
    d1: f64,
    // geodesic distance from the source to the pseudosource (called sigma in the review)
    sigma: f64,
    // source id
    s: usize,
    // pseudosource id
    p: usize,
}

/// Structure representing the ICH algorithm.
pub struct ICH {
    manifold: Manifold,
    source_vertices: Vec<usize>,
    source_points: Vec<(usize, Point3<f64>)>,
    kept_faces: Vec<usize>,
}

impl ICH {
    /// Creates a new ICH instance.
    pub fn new(
        manifold: Manifold,
        source_vertices: Vec<usize>,
        source_points: Vec<(usize, Point3<f64>)>,
        kept_faces: Vec<usize>,
    ) -> Self {
        ICH {
            manifold,
            source_vertices,
            source_points,
            kept_faces,
        }
    }

    /// Runs the ICH algorithm.
    pub fn run(&self) {
        self.init();
        unimplemented!()
    }
}

impl ICH {
    fn init(&self) {
        for _ in &self.source_vertices {
            // Initialize windows at source vertices
        }

        for (_, _) in &self.source_points {
            // Initialize windows at source points
        }

        unimplemented!()
    }

    fn propagate_window(&self, window: &Window) {
        unimplemented!()
    }
}
