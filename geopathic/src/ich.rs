//! Implementation of the Improved Chen-Han (ICH) algorithm for pathfinding.

use nalgebra::Point3;
use std::collections::BinaryHeap;

use crate::mesh::Mesh;

#[derive(Debug, Clone)]
struct Window {
    // index of the edge the window is on
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
    // minimum distance within this window
    min_distance: f64,
}

impl Window {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        edge: usize,
        b0: f64,
        b1: f64,
        d0: f64,
        d1: f64,
        sigma: f64,
        s: usize,
        p: usize,
    ) -> Self {
        let mut win = Window {
            edge,
            b0,
            b1,
            d0,
            d1,
            sigma,
            s,
            p,
            min_distance: 0.0,
        };
        win.compute_min_distance();
        win
    }

    fn compute_min_distance(&mut self) {
        let tau = self.b1 - self.b0;
        let x_projection = (self.d0.powi(2) - self.d1.powi(2) + tau.powi(2)) / (2.0 * tau);
        if x_projection < 0.0 {
            self.min_distance = self.d0 + self.sigma;
        } else if x_projection > tau {
            self.min_distance = self.d1 + self.sigma;
        } else {
            let y_projection = f64::sqrt(self.d0.powi(2) - x_projection.powi(2));
            self.min_distance = y_projection + self.sigma;
        }
    }
}

impl PartialEq for Window {
    fn eq(&self, other: &Self) -> bool {
        self.min_distance == other.min_distance
    }
}

impl PartialOrd for Window {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Window {}

impl Ord for Window {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.min_distance > other.min_distance {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }
}

#[derive(Debug, Clone)]
pub struct PseudoWindow {}

impl PartialEq for PseudoWindow {
    fn eq(&self, _other: &Self) -> bool {
        true // FIXME:
    }
}

impl PartialOrd for PseudoWindow {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for PseudoWindow {}

impl Ord for PseudoWindow {
    fn cmp(&self, _other: &Self) -> std::cmp::Ordering {
        std::cmp::Ordering::Equal // FIXME:
    }
}

/// Structure representing the ICH algorithm.
pub struct ICH {
    mesh: Mesh,
    source_vertices: Vec<usize>,
    source_points: Vec<(usize, Point3<f64>)>,
    kept_faces: Vec<usize>,
    window_queue: BinaryHeap<Window>,
    pseudo_source_queue: BinaryHeap<PseudoWindow>,
}

impl ICH {
    /// Creates a new ICH instance.
    pub fn new(
        mesh: Mesh,
        source_vertices: Vec<usize>,
        source_points: Vec<(usize, Point3<f64>)>,
        kept_faces: Vec<usize>,
    ) -> Self {
        ICH {
            mesh,
            source_vertices,
            source_points,
            kept_faces,
            window_queue: BinaryHeap::new(),
            pseudo_source_queue: BinaryHeap::new(),
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
        for source in &self.source_vertices {
            for edge in self.mesh.edges_of_vertex(*source) {
                let next_edge_id = edge.next_edge;
                let next_edge = &self.mesh.edges[next_edge_id];

                let win = Window::new(
                    next_edge_id,
                    0.0,
                    next_edge.length,
                    edge.length,
                    next_edge.length,
                    0.0,
                    *source,
                    *source,
                );
            }
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
