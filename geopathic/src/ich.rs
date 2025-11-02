//! Implementation of the Improved Chen-Han (ICH) algorithm for pathfinding.

use nalgebra::Point3;
use std::collections::BinaryHeap;

use crate::mesh::Mesh;

#[derive(Debug, Clone)]
struct Window {
    /// index of the edge the window is on
    edge: usize,
    /// first endpoint of the edge
    b0: f64,
    /// second endpoint of the edge
    b1: f64,
    /// distance from the first endpoint to the pseudosource
    d0: f64,
    /// distance from the second endpoint to the pseudosource
    d1: f64,
    /// geodesic distance from the source to the pseudosource (called sigma in the review)
    sigma: f64,
    /// source id
    s: usize,
    /// pseudosource id
    p: usize,
    /// minimum distance within this window
    min_distance: f64,
    /// birth time of the window
    birth_time: usize,
    /// level of the window
    level: usize,
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
            birth_time: 0,
            level: 0,
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

#[derive(Debug, Clone, PartialEq)]
pub struct PseudoWindow {
    vertex: usize,
    distance: f64,
    s: usize,
    p: usize,
    birth_time: usize,
    level: usize,
}

impl PartialOrd for PseudoWindow {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for PseudoWindow {}

impl Ord for PseudoWindow {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.distance < other.distance {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
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
    stats: ICHStats,
    vertex_infos: Vec<VertexInfo>
}

impl ICH {
    /// Creates a new ICH instance.
    pub fn new(
        mesh: Mesh,
        source_vertices: Vec<usize>,
        source_points: Vec<(usize, Point3<f64>)>,
        kept_faces: Vec<usize>,
    ) -> Self {
        let len = mesh.vertices.len();
        ICH {
            mesh,
            source_vertices,
            source_points,
            kept_faces,
            window_queue: BinaryHeap::new(),
            pseudo_source_queue: BinaryHeap::new(),
            stats: ICHStats::default(),
            vertex_infos: vec![VertexInfo::new(); len],
        }
    }

    /// Runs the ICH algorithm.
    pub fn run(&mut self) {
        self.init();
        unimplemented!()
    }
}

impl ICH {
    fn init(&mut self) {
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
                self.window_queue.push(win);
                self.stats.window_created();

                let opposite_vertex_id = edge.end;
                if edge.length < self.vertex_infos[opposite_vertex_id].distance {
                    self.vertex_infos[opposite_vertex_id].birth_time = 0;
                    self.vertex_infos[opposite_vertex_id].distance = edge.length;
                    self.vertex_infos[opposite_vertex_id].enter_edge = next_edge_id;
                    self.vertex_infos[opposite_vertex_id].p = *source;
                    self.vertex_infos[opposite_vertex_id].s = *source;

                    if self.mesh.angles[opposite_vertex_id] < 2.0 * std::f64::consts::PI {
                        continue
                    }

                    let pseudo_win = PseudoWindow {
                        vertex: opposite_vertex_id,
                        distance: edge.length,
                        s: *source,
                        p: *source,
                        birth_time: self.vertex_infos[opposite_vertex_id].birth_time,
                        level: 0,
                    };
                    self.pseudo_source_queue.push(pseudo_win);
                    self.stats.pseudo_source_created();
                }
            }
            self.vertex_infos[*source].birth_time = 0;
            self.vertex_infos[*source].distance = 0.0;
            self.vertex_infos[*source].enter_edge = usize::MAX;
            self.vertex_infos[*source].is_source = true;
            self.vertex_infos[*source].s = *source;
            self.vertex_infos[*source].p = *source;
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

#[derive(Debug, Default, Clone)]
pub struct ICHStats {
    pub windows_created: usize,
    pub windows_propagated: usize,
    pub pseudo_sources_created: usize,
    pub pseudo_sources_propagated: usize,
}

impl ICHStats {
    pub fn window_created(&mut self) {
        self.windows_created += 1;
    }

    pub fn window_propagated(&mut self) {
        self.windows_propagated += 1;
    }

    pub fn pseudo_source_created(&mut self) {
        self.pseudo_sources_created += 1;
    }

    pub fn pseudo_source_propagated(&mut self) {
        self.pseudo_sources_propagated += 1;
    }
}

#[derive(Debug, Clone)]
struct VertexInfo {
    birth_time: usize,
    distance: f64,
    enter_edge: usize,
    is_source: bool,
    p: usize,
    s: usize
}

impl VertexInfo {
    pub fn new() -> Self {
        VertexInfo {
            birth_time: usize::MAX,
            distance: f64::INFINITY,
            enter_edge: usize::MAX,
            is_source: false,
            p: usize::MAX,
            s: usize::MAX
        }
    }
}