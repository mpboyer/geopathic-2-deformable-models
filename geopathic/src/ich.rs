//! Implementation of the Improved Chen-Han (ICH) algorithm for pathfinding.

use nalgebra::Point3;
use std::collections::BinaryHeap;

use crate::mesh::{Mesh, dist};

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
    /// Creates a new Window and computes its minimum distance.
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

/// Similar to Window, but for pseudo sources.
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
    /// The mesh on which to run the algorithm.
    mesh: Mesh,
    /// The vertices to use as sources.
    source_vertices: Vec<usize>,
    /// The non-vertices points to use as sources (face index and position).
    source_points: Vec<(usize, Point3<f64>)>,
    /// The faces to keep during the algorithm.
    kept_faces: Vec<usize>,
    /// The windows stored for distance reconstruction.
    stored_windows: Vec<Window>,
    /// The window queue for processing.
    window_queue: BinaryHeap<Window>,
    /// The pseudo source queue for processing.
    pseudo_source_queue: BinaryHeap<PseudoWindow>,
    /// Statistics collected during the algorithm.
    stats: ICHStats,
    /// Information about each vertex.
    vertex_infos: Vec<VertexInfo>,
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
            stored_windows: Vec::new(),
            window_queue: BinaryHeap::new(),
            pseudo_source_queue: BinaryHeap::new(),
            stats: ICHStats::default(),
            vertex_infos: vec![VertexInfo::new(); len],
        }
    }

    /// Runs the ICH algorithm.
    pub fn run(&mut self) {
        self.init();

        // main loop: while there are windows or pseudo windows to process
        while !self.window_queue.is_empty() || !self.pseudo_source_queue.is_empty() {
            self.stats.update_max_queue_size(self.window_queue.len());
            self.stats
                .update_max_pseudo_queue_size(self.pseudo_source_queue.len());

            // remove invalid windows, that is, whose birth time is not equal than the current one
            while let Some(win) = self.window_queue.peek() {
                if win.birth_time != self.vertex_infos[win.p].birth_time {
                    self.window_queue.pop();
                } else {
                    break;
                }
            }

            // remove invalid pseudo windows
            while let Some(pseudo_win) = self.pseudo_source_queue.peek() {
                if pseudo_win.birth_time != self.vertex_infos[pseudo_win.vertex].birth_time {
                    self.pseudo_source_queue.pop();
                } else {
                    break;
                }
            }

            // if there is a window with smaller min_distance than the smallest pseudo window distance, propagate it
            if !self.window_queue.is_empty()
                && (self.pseudo_source_queue.is_empty()
                    || self.window_queue.peek().unwrap().min_distance
                        <= self.pseudo_source_queue.peek().unwrap().distance)
            {
                let win = self.window_queue.pop().unwrap();
                if win.level > self.mesh.faces.len() {
                    continue;
                }

                // we save the window to reconstruct the geodesic distances later
                if let Some(twin_edge_id) = self.mesh.edges[win.edge].twin_edge {
                    let twin_edge = &self.mesh.edges[twin_edge_id];
                    if self.kept_faces.contains(&twin_edge.face) {
                        self.stored_windows.push(win);
                    }
                }

                self.propagate_window(self.stored_windows.last().unwrap());
            }
            // else if there is a pseudo window with smaller distance than the smallest window min_distance, generate sub-windows from it
            else if !self.pseudo_source_queue.is_empty()
                && (self.window_queue.is_empty()
                    || self.window_queue.peek().unwrap().min_distance
                        >= self.pseudo_source_queue.peek().unwrap().distance)
            {
                let pseudo_win = self.pseudo_source_queue.pop().unwrap();
                if pseudo_win.level > self.mesh.faces.len() {
                    continue;
                }
                self.generate_sub_windows(&pseudo_win);
            }
        }
    }
}

impl ICH {
    /// Initializes the ICH algorithm by creating initial windows from source vertices and points.
    fn init(&mut self) {
        // create initial windows from source vertices
        for source in &self.source_vertices {
            for edge in self.mesh.edges_of_vertex(*source) {
                let next_edge_id = edge.next_edge;
                let next_edge = &self.mesh.edges[next_edge_id];

                // create a new window on the next edge
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
                        continue;
                    }

                    // create a pseudo window for the opposite vertex
                    let pseudo_win = PseudoWindow {
                        vertex: opposite_vertex_id,
                        distance: edge.length,
                        s: *source,
                        p: *source,
                        birth_time: self.vertex_infos[opposite_vertex_id].birth_time,
                        level: 0,
                    };
                    self.pseudo_source_queue.push(pseudo_win);
                }
            }

            // set source vertex info
            self.vertex_infos[*source].birth_time = 0;
            self.vertex_infos[*source].distance = 0.0;
            self.vertex_infos[*source].enter_edge = usize::MAX;
            self.vertex_infos[*source].is_source = true;
            self.vertex_infos[*source].s = *source;
            self.vertex_infos[*source].p = *source;
        }

        // create initial windows from source points (non-vertices)
        for (i, (face_id, position)) in self.source_points.iter().enumerate() {
            for j in 0..3 {
                let opposite_edge_id = self.mesh.faces[*face_id].edges[j];
                let opposite_edge = &self.mesh.edges[opposite_edge_id];

                // create a new "vertex id" for the source point (which is not a vertex)
                let source_id = self.mesh.vertices.len() + i;

                // create a new window on the opposite edge
                let win = Window::new(
                    opposite_edge_id,
                    0.0,
                    opposite_edge.length,
                    dist(position, &self.mesh.vertices[opposite_edge.start].position),
                    dist(position, &self.mesh.vertices[opposite_edge.end].position),
                    0.0,
                    source_id,
                    source_id,
                );
                self.window_queue.push(win);
                self.stats.window_created();

                let opposite_vertex_id = opposite_edge.start;
                if dist(position, &self.mesh.vertices[opposite_vertex_id].position)
                    > self.vertex_infos[opposite_vertex_id].distance
                {
                    continue;
                }

                // set vertex info
                self.vertex_infos[opposite_vertex_id].birth_time = 0;
                self.vertex_infos[opposite_vertex_id].distance =
                    dist(position, &self.mesh.vertices[opposite_vertex_id].position);
                self.vertex_infos[opposite_vertex_id].enter_edge = usize::MAX;
                self.vertex_infos[opposite_vertex_id].s = source_id;
                self.vertex_infos[opposite_vertex_id].p = source_id;

                if self.mesh.angles[opposite_vertex_id] < 2.0 * std::f64::consts::PI {
                    continue;
                }

                // create a pseudo window for the opposite vertex
                let pseudo_win = PseudoWindow {
                    vertex: opposite_vertex_id,
                    distance: dist(position, &self.mesh.vertices[opposite_vertex_id].position),
                    s: source_id,
                    p: source_id,
                    birth_time: self.vertex_infos[opposite_vertex_id].birth_time,
                    level: 0,
                };
                self.pseudo_source_queue.push(pseudo_win);
            }
        }
    }

    fn propagate_window(&self, window: &Window) {
        unimplemented!()
    }

    fn generate_sub_windows(&self, pseudo_window: &PseudoWindow) {
        unimplemented!()
    }
}

#[derive(Debug, Default, Clone)]
pub struct ICHStats {
    pub windows_created: usize,
    pub windows_propagated: usize,
    pub max_queue_size: usize,
    pub max_pseudo_queue_size: usize,
}

impl ICHStats {
    pub fn window_created(&mut self) {
        self.windows_created += 1;
    }

    pub fn window_propagated(&mut self) {
        self.windows_propagated += 1;
    }

    pub fn update_max_queue_size(&mut self, size: usize) {
        if size > self.max_queue_size {
            self.max_queue_size = size;
        }
    }

    pub fn update_max_pseudo_queue_size(&mut self, size: usize) {
        if size > self.max_pseudo_queue_size {
            self.max_pseudo_queue_size = size;
        }
    }
}

/// Information about each vertex during the ICH algorithm.
#[derive(Debug, Clone)]
struct VertexInfo {
    birth_time: usize,
    distance: f64,
    enter_edge: usize,
    is_source: bool,
    p: usize,
    s: usize,
}

impl VertexInfo {
    pub fn new() -> Self {
        VertexInfo {
            birth_time: usize::MAX,
            distance: f64::INFINITY,
            enter_edge: usize::MAX,
            is_source: false,
            p: usize::MAX,
            s: usize::MAX,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::load_manifold;

    #[test]
    fn test_ich_init_teddy() {
        let manifold = load_manifold("../examples/models/teddy.obj").unwrap();
        let mesh = Mesh::from_manifold(&manifold);
        let source_vertices = vec![0, 1, 2, 3, 4];
        let source_points = vec![];
        let kept_faces = vec![];

        let mut ich = ICH::new(mesh, source_vertices, source_points, kept_faces);
        ich.init();

        assert_eq!(ich.stats.windows_created, 32);
    }

    #[test]
    fn test_ich_init_teddy_2() {
        let manifold = load_manifold("../examples/models/teddy.obj").unwrap();
        let mesh = Mesh::from_manifold(&manifold);
        let source_vertices = vec![5, 6, 7, 8, 9];
        let source_points = vec![
            (0, Point3::new(0.0, 0.0, 0.0)),
            (1, Point3::new(1.0, 1.0, 1.0)),
            (2, Point3::new(2.0, 2.0, 2.0)),
        ];
        let kept_faces = vec![];

        let mut ich = ICH::new(mesh, source_vertices, source_points, kept_faces);
        ich.init();

        assert_eq!(ich.stats.windows_created, 43);
    }
}
