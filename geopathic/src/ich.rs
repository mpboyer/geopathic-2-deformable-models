//! Implementation of the Improved Chen-Han (ICH) algorithm for pathfinding.

use crate::mesh::{Mesh, dist};
use nalgebra::{Point2, Point3};
use std::collections::BinaryHeap;

const RELATIVE_ERROR: f64 = 1e-6;

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
    birth_time: Option<usize>,
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
            birth_time: Some(0),
            level: 0,
        };
        win.compute_min_distance();
        win
    }

    pub fn build(
        parent: &Window,
        edge: usize,
        t0: f64,
        t1: f64,
        v_start: Point2<f64>,
        v_end: Point2<f64>,
    ) -> Self {
        unimplemented!()
    }

    /// Computes the minimum distance within the window.
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

    /// Flattens the source point onto a 2D plane for calculations.
    pub fn flatten_source(&self) -> Point2<f64> {
        let tau = self.b1 - self.b0;
        let x_projection = (self.d0.powi(2) - self.d1.powi(2) + tau.powi(2)) / (2.0 * tau);
        let y_projection = f64::sqrt((self.d0.powi(2) - x_projection.powi(2)).abs());
        Point2::new(x_projection + self.b0, y_projection)
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
    birth_time: Option<usize>,
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
    /// Information for split checks.
    split_infos: Vec<SplitInfo>,
}

impl ICH {
    /// Creates a new ICH instance.
    pub fn new(
        mesh: Mesh,
        source_vertices: Vec<usize>,
        source_points: Vec<(usize, Point3<f64>)>,
        kept_faces: Vec<usize>,
    ) -> Self {
        let nb_vertices = mesh.vertices.len();
        let nb_edges = mesh.edges.len();
        ICH {
            mesh,
            source_vertices,
            source_points,
            kept_faces,
            stored_windows: Vec::new(),
            window_queue: BinaryHeap::new(),
            pseudo_source_queue: BinaryHeap::new(),
            stats: ICHStats::default(),
            vertex_infos: vec![VertexInfo::new(); nb_vertices],
            split_infos: vec![SplitInfo::new(); nb_edges],
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
                        self.stored_windows.push(win.clone());
                    }
                }

                self.propagate_window(&win);
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
                    self.vertex_infos[opposite_vertex_id].birth_time = Some(0);
                    self.vertex_infos[opposite_vertex_id].distance = edge.length;
                    self.vertex_infos[opposite_vertex_id].enter_edge = Some(next_edge_id);
                    self.vertex_infos[opposite_vertex_id].p = Some(*source);
                    self.vertex_infos[opposite_vertex_id].s = Some(*source);

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
            self.vertex_infos[*source].birth_time = Some(0);
            self.vertex_infos[*source].distance = 0.0;
            self.vertex_infos[*source].enter_edge = None;
            self.vertex_infos[*source].is_source = true;
            self.vertex_infos[*source].s = Some(*source);
            self.vertex_infos[*source].p = Some(*source);
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
                self.vertex_infos[opposite_vertex_id].birth_time = Some(0);
                self.vertex_infos[opposite_vertex_id].distance =
                    dist(position, &self.mesh.vertices[opposite_vertex_id].position);
                self.vertex_infos[opposite_vertex_id].enter_edge = None;
                self.vertex_infos[opposite_vertex_id].s = Some(source_id);
                self.vertex_infos[opposite_vertex_id].p = Some(source_id);

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

    fn propagate_window(&mut self, window: &Window) {
        // extract the three edges of the face opposite to the window's edge
        let e0 = self.mesh.edges[window.edge].twin_edge;
        let e0 = match e0 {
            Some(id) => id,
            None => return,
        };
        let e1 = self.mesh.edges[e0].next_edge;
        let e2 = self.mesh.edges[e1].next_edge;

        // flatten the source point of the window
        let source_2d = window.flatten_source();

        //
        let left = Point2::new(window.b0, 0.0);
        let right = Point2::new(window.b1, 0.0);
        let l0 = self.mesh.edges[e0].length;
        let l1 = self.mesh.edges[e1].length;
        let l2 = self.mesh.edges[e2].length;
        let v0 = Point2::new(0.0, 0.0);
        let v1 = Point2::new(l1, 0.0);
        let x = (l1.powi(2) + l0.powi(2) - l2.powi(2)) / (2.0 * l0);
        let v2 = Point2::new(x, -f64::sqrt((l1.powi(2) - x.powi(2)).abs()));

        let inter_x = v2.x - v2.y * (v2.x - source_2d.x) / (v2.y - source_2d.y);

        // only right child window
        let (left_win, right_win) = if inter_x <= left.x {
            // compute the window
            let t0 = self.intersect(source_2d, left, v2, v1);
            let t1 = self.intersect(source_2d, right, v2, v1);
            let right_win = Window::build(window, e2, t0, t1, v2, v1);

            // check if it is valid
            let right_win = if self.is_valid_window(window, false) {
                Some(right_win)
            } else {
                None
            };

            // return only right window
            (None, right_win)
        }
        // only left child window
        else if inter_x >= right.x {
            // compute the window
            let t0 = self.intersect(source_2d, left, v2, v1);
            let t1 = self.intersect(source_2d, right, v2, v1);
            let left_win = Window::build(window, e1, t0, t1, v0, v2);

            // check if it is valid
            let left_win = if self.is_valid_window(window, true) {
                Some(left_win)
            } else {
                None
            };

            // return only left window
            (left_win, None)
        }
        // both child windows
        else {
            let opposite_vertex = self.mesh.edges[e1].end;
            let direct_distance = (v2 - source_2d).norm();

            // "one angle, one split" rule
            let (build_left, build_right) = if direct_distance + window.sigma > self.split_infos[e0].distance && (direct_distance + window.sigma) / self.split_infos[e0].distance - 1.0 > RELATIVE_ERROR {
                (self.split_infos[e0].x < inter_x, self.split_infos[e0].x >= inter_x)
            } else {
                if direct_distance + window.sigma < self.split_infos[e0].distance {
                    self.split_infos[e0].distance = direct_distance + window.sigma;
                    self.split_infos[e0].s = Some(window.s);
                    self.split_infos[e0].p = Some(window.p);
                    self.split_infos[e0].level = Some(window.level);
                    self.split_infos[e0].x = l0 - inter_x;
                }

                if direct_distance + window.sigma < self.vertex_infos[opposite_vertex].distance {
                    let birth = self.vertex_infos[window.p].birth_time;
                    self.vertex_infos[opposite_vertex].birth_time = match birth {
                        Some(t) => Some(t + 1),
                        None => Some(0),
                    };

                    self.vertex_infos[opposite_vertex].distance = direct_distance + window.sigma;
                    self.vertex_infos[opposite_vertex].enter_edge = Some(e0);
                    self.vertex_infos[opposite_vertex].s = Some(window.s);
                    self.vertex_infos[opposite_vertex].p = Some(window.p);

                    if self.mesh.angles[opposite_vertex] > 2.0 * std::f64::consts::PI {
                        let pseudo_win = PseudoWindow {
                            vertex: opposite_vertex,
                            distance: self.vertex_infos[opposite_vertex].distance,
                            s: window.s,
                            p: window.p,
                            birth_time: self.vertex_infos[opposite_vertex].birth_time,
                            level: window.level + 1,
                        };
                        self.pseudo_source_queue.push(pseudo_win);
                    }
                }

                (true, true)
            };

            // compute the windows
            let left_child = if build_left {
                let t0 = self.intersect(source_2d, left, v0, v2);
                Some(Window::build(window, e1, t0, 0.0, v0, v2))
            } else {
                None
            };
            let right_child = if build_right {
                let t1 = self.intersect(source_2d, right, v2, v1);
                Some(Window::build(window, e2, 0.0, t1, v2, v1))
            } else {
                None
            };

            (left_child, right_child)
        };

        if let Some(win) = left_win {
            self.window_queue.push(win);
            self.stats.window_created();
        }
        if let Some(win) = right_win {
            self.window_queue.push(win);
            self.stats.window_created();
        }
    }

    /// Generates sub-windows from a pseudo window.
    fn generate_sub_windows(&self, pseudo_window: &PseudoWindow) {
        unimplemented!()
    }

    /// Computes the intersection point between two lines defined by points.
    fn intersect(&self, v0: Point2<f64>, v1: Point2<f64>, p0: Point2<f64>, p1: Point2<f64>) -> f64 {
        let a00 = p0.x - p1.x;
        let a01 = v1.x - v0.x;
        let a10 = p0.y - p1.y;
        let a11 = v1.y - v0.y;
        let b0 = v1.x - p1.x;
        let b1 = v1.y - p1.y;

        let det = a00 * a11 - a01 * a10;
        (b0 * a11 - b1 * a01) / det
    }

    /// Checks if a window is valid based on ICH's filters.
    fn is_valid_window(&self, window: &Window, is_left: bool) -> bool {
        // degenerate window
        if window.b1 <= window.b0 {
            return false;
        }

        // vertices
        let v1 = self.mesh.edges[window.edge].start;
        let v2 = self.mesh.edges[window.edge].end;
        let v3 = self.mesh.edges[self.mesh.edges[window.edge].next_edge].end;

        // lengths of the edges
        let l0 = self.mesh.edges[window.edge].length;
        let l1 = self.mesh.edges[self.mesh.edges[window.edge].next_edge].length;
        let l2 = self.mesh.edges[self.mesh.edges[self.mesh.edges[window.edge].next_edge].next_edge]
            .length;

        // compute point p3 in 2D
        let x = (l2.powi(2) + l0.powi(2) - l1.powi(2)) / (2.0 * l0);
        let p3 = Point2::new(x, f64::sqrt((l1.powi(2) - x.powi(2)).abs()));

        // window points in 2D
        let a = Point2::new(window.b0, 0.0);
        let b = Point2::new(window.b1, 0.0);
        let src_2d = window.flatten_source();

        // precompute some distances
        let s_a = window.sigma + (src_2d - a).norm();
        let s_b = window.sigma + (src_2d - b).norm();
        let p3_a = self.vertex_infos[v3].distance + (p3 - a).norm();
        let p3_b = self.vertex_infos[v3].distance + (p3 - b).norm();

        // apply the filters
        if s_b > self.vertex_infos[v1].distance + window.b1
            && s_b / (self.vertex_infos[v1].distance + window.b1) - 1.0 > 0.0
        {
            return false;
        }
        if s_a > self.vertex_infos[v2].distance + (l0 - window.b0)
            && s_a / (self.vertex_infos[v2].distance + (l0 - window.b0)) - 1.0 > 0.0
        {
            return false;
        }
        if is_left && s_a > p3_a && s_a / p3_a - 1.0 > 0.0 {
            return false;
        }
        if !is_left && s_b > p3_b && s_b / p3_b - 1.0 > RELATIVE_ERROR {
            return false;
        }

        true
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
    birth_time: Option<usize>,
    distance: f64,
    enter_edge: Option<usize>,
    is_source: bool,
    p: Option<usize>,
    s: Option<usize>,
}

impl VertexInfo {
    pub fn new() -> Self {
        VertexInfo {
            birth_time: None,
            distance: f64::INFINITY,
            enter_edge: None,
            is_source: false,
            p: None,
            s: None,
        }
    }
}

/// Information for split checks during window propagation.
#[derive(Debug, Clone)]
struct SplitInfo {
    distance: f64,
    s: Option<usize>,
    p: Option<usize>,
    level: Option<usize>,
    x: f64,
}

impl SplitInfo {
    pub fn new() -> Self {
        SplitInfo {
            distance: f64::INFINITY,
            s: None,
            p: None,
            level: None,
            x: f64::INFINITY,
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
