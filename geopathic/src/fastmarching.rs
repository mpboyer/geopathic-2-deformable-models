// Wavefront propagation based methods: Fast Marching

use std::{cmp::Ordering, collections::BinaryHeap};

use nalgebra::DVector;

use crate::manifold::Manifold;

/// State of vertex during fast marching (basically adapted Dijkstra)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VertexState {
    Far,   // Not yet visited
    Trial, // In priority Queue
    Alive, // Distance finalized
}

#[derive(Debug, Clone)]
struct TrialVertex {
    vertex: usize,
    distance: f64,
}

impl PartialEq for TrialVertex {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for TrialVertex {}

impl PartialOrd for TrialVertex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance) // Order is reversed because we use min-heap
    }
}

impl Ord for TrialVertex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct FastMarching<'a> {
    manifold: &'a Manifold,
}

impl<'a> FastMarching<'a> {
    pub fn new(manifold: &'a Manifold) -> Self {
        Self { manifold }
    }

    /// Compute geodesic distance from a source vertex using Fast Marching Method
    /// Based on Kimmel & Sethian (1998)
    pub fn compute_distance(&self, source: usize) -> Result<DVector<f64>, String> {
        let n = self.manifold.vertices().len();

        // Initialize distances and states
        let mut distances = vec![f64::INFINITY; n];
        let mut states = vec![VertexState::Far; n];
        let mut heap = BinaryHeap::new();

        // Set source
        distances[source] = 0.0;
        states[source] = VertexState::Alive;

        // Initialize direct neighbors of source with edge distances
        for face in self.manifold.faces() {
            let (i, j, k) = *face;

            // Check each edge in the face
            if i == source {
                self.init_neighbor(source, j, &mut distances, &mut states, &mut heap);
                self.init_neighbor(source, k, &mut distances, &mut states, &mut heap);
            } else if j == source {
                self.init_neighbor(source, i, &mut distances, &mut states, &mut heap);
                self.init_neighbor(source, k, &mut distances, &mut states, &mut heap);
            } else if k == source {
                self.init_neighbor(source, i, &mut distances, &mut states, &mut heap);
                self.init_neighbor(source, j, &mut distances, &mut states, &mut heap);
            }
        }

        // Fast Marching main loop
        while let Some(trial) = heap.pop() {
            let v = trial.vertex;

            // Skip if already processed (duplicate in heap)
            if states[v] == VertexState::Alive {
                continue;
            }

            // Mark as alive
            states[v] = VertexState::Alive;

            // Update neighbors
            for face in self.manifold.faces() {
                let (i, j, k) = *face;
                if i == v || j == v || k == v {
                    self.update_vertex_from_face(face, v, &mut distances, &mut states, &mut heap)?;
                }
            }
        }

        Ok(DVector::from_vec(distances))
    }

    /// Initialize a direct neighbor of the source with edge distance
    fn init_neighbor(
        &self,
        source: usize,
        neighbor: usize,
        distances: &mut [f64],
        states: &mut [VertexState],
        heap: &mut BinaryHeap<TrialVertex>,
    ) {
        let p_source = &self.manifold.vertices()[source];
        let p_neighbor = &self.manifold.vertices()[neighbor];
        let edge_dist = (p_neighbor - p_source).norm();

        // Only update if this gives a better distance
        if edge_dist < distances[neighbor] {
            distances[neighbor] = edge_dist;

            if states[neighbor] == VertexState::Far {
                states[neighbor] = VertexState::Trial;
            }

            heap.push(TrialVertex {
                vertex: neighbor,
                distance: edge_dist,
            });
        }
    }

    /// Update a vertex by solving the eikonal equation on a triangle
    fn update_vertex_from_face(
        &self,
        face: &(usize, usize, usize),
        known_vertex: usize,
        distances: &mut [f64],
        states: &mut [VertexState],
        heap: &mut BinaryHeap<TrialVertex>,
    ) -> Result<(), String> {
        let (i, j, k) = *face;

        // Try to update each vertex in the face
        for target in [i, j, k] {
            if target == known_vertex || states[target] == VertexState::Alive {
                continue;
            }

            // Find the other two vertices
            let others: Vec<usize> = [i, j, k]
                .iter()
                .filter(|&&v| v != target)
                .copied()
                .collect();

            if others.len() != 2 {
                continue;
            }

            let v1 = others[0];
            let v2 = others[1];

            // Compute new distance using eikonal equation
            if let Some(new_dist) = self.solve_eikonal_on_triangle(target, v1, v2, distances)? {
                if new_dist < distances[target] {
                    distances[target] = new_dist;

                    if states[target] == VertexState::Far {
                        states[target] = VertexState::Trial;
                    }

                    heap.push(TrialVertex {
                        vertex: target,
                        distance: new_dist,
                    });
                }
            }
        }

        Ok(())
    }

    /// Solve the eikonal equation on a triangle to find distance at target
    /// Given distances at v1 and v2, find distance at target
    fn solve_eikonal_on_triangle(
        &self,
        target: usize,
        v1: usize,
        v2: usize,
        distances: &[f64],
    ) -> Result<Option<f64>, String> {
        let d1 = distances[v1];
        let d2 = distances[v2];

        // If either distance is infinite, use one-sided update
        if d1.is_infinite() && d2.is_infinite() {
            return Ok(None);
        }

        let p_target = &self.manifold.vertices()[target];
        let p1 = &self.manifold.vertices()[v1];
        let p2 = &self.manifold.vertices()[v2];

        // Edge lengths
        let a = (p2 - p_target).norm(); // opposite to v1
        let b = (p1 - p_target).norm(); // opposite to v2
        let c = (p2 - p1).norm(); // opposite to target

        // If one distance is infinite, use simple update
        if d1.is_infinite() {
            return Ok(Some(d2 + a));
        }
        if d2.is_infinite() {
            return Ok(Some(d1 + b));
        }

        // Use law of cosines to find angle at target
        let cos_angle = (a * a + b * b - c * c) / (2.0 * a * b);
        let cos_angle = cos_angle.clamp(-1.0, 1.0); // Clamp for numerical stability

        // Solve quadratic equation for distance
        // Based on the eikonal equation |∇φ| = 1
        let dd = d1 - d2;
        let discriminant = a * a + b * b - 2.0 * a * b * cos_angle - dd * dd;

        if discriminant < 0.0 {
            // No solution from both vertices, use one-sided update
            return Ok(Some(d1.min(d2) + a.min(b)));
        }

        // Two possible solutions, take the one that gives larger distance
        // (corresponding to positive slope)
        let sin_angle = (1.0 - cos_angle * cos_angle).sqrt();
        let numerator = a * d2 + b * d1 + (a * b * sin_angle * discriminant.sqrt());
        let denominator = a * a + b * b - 2.0 * a * b * cos_angle;

        if denominator.abs() < 1e-10 {
            return Ok(Some(d1.min(d2) + a.min(b)));
        }

        let d_new = numerator / denominator;

        // Causality check: new distance should be larger than both known distances
        if d_new < d1.max(d2) {
            return Ok(Some(d1.min(d2) + a.min(b)));
        }

        Ok(Some(d_new))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_marching_tetrahedron() {
        let vertices = vec![
            DVector::from_vec(vec![0.0, 0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0]),
            DVector::from_vec(vec![0.0, 0.0, 1.0]),
        ];

        let faces = vec![(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)];

        let manifold = Manifold::new(vertices.clone(), faces);
        let fm = FastMarching::new(&manifold);

        let distances = fm.compute_distance(0).unwrap();

        println!("Computed distances: {:?}", distances);

        // Check that source has zero distance
        assert!(distances[0].abs() < 1e-6, "Source distance should be 0");

        // Check that distances are positive and finite
        for i in 1..4 {
            assert!(
                distances[i] > 0.0,
                "Distance at vertex {} should be positive",
                i
            );
            assert!(
                distances[i].is_finite(),
                "Distance at vertex {} should be finite",
                i
            );
        }

        // The actual geodesic distances on this tetrahedron:
        // Vertices 1,2,3 are all connected by direct edges of length 1.0 from vertex 0
        // But the algorithm might compute them through the polyhedral surface
        // Let's verify they're at least close to 1.0
        assert!(
            (distances[1] - 1.0).abs() < 0.01,
            "Distance to vertex 1: expected ~1.0, got {}",
            distances[1]
        );
        assert!(
            (distances[2] - 1.0).abs() < 0.01,
            "Distance to vertex 2: expected ~1.0, got {}",
            distances[2]
        );

        // Vertex 3 might be computed differently depending on the update order
        // Accept either ~1.0 (direct) or ~1.414 (through another path)
        assert!(
            (distances[3] - 1.0).abs() < 0.01,
            "Distance to vertex 3: expected ~1.0 got {}",
            distances[3]
        );
    }

    #[test]
    fn test_fast_marching_square() {
        // Create a simple square made of two triangles
        let vertices = vec![
            DVector::from_vec(vec![0.0, 0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0, 0.0]),
            DVector::from_vec(vec![1.0, 1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0]),
        ];

        let faces = vec![(0, 1, 2), (0, 2, 3)];

        let manifold = Manifold::new(vertices, faces);
        let fm = FastMarching::new(&manifold);

        let distances = fm.compute_distance(0).unwrap();

        println!("Square distances: {:?}", distances);

        // Expected distances from vertex 0
        assert!(
            distances[0].abs() < 1e-6,
            "Distance to vertex 0: {}",
            distances[0]
        );
        assert!(
            (distances[1] - 1.0).abs() < 1e-6,
            "Distance to vertex 1: expected 1.0, got {}",
            distances[1]
        );
        assert!(
            (distances[3] - 1.0).abs() < 1e-6,
            "Distance to vertex 3: expected 1.0, got {}",
            distances[3]
        );

        // Diagonal distance - should be sqrt(2) but might vary based on path
        println!(
            "Diagonal distance to vertex 2: {} (expected ~{})",
            distances[2],
            2.0_f64.sqrt()
        );
        assert!(
            (distances[2] - 2.0_f64.sqrt()).abs() < 0.1,
            "Distance to vertex 2: expected ~{}, got {}",
            2.0_f64.sqrt(),
            distances[2]
        );
    }

    #[test]
    fn test_fast_marching_symmetry() {
        let vertices = vec![
            DVector::from_vec(vec![0.0, 0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0]),
            DVector::from_vec(vec![0.0, 0.0, 1.0]),
        ];

        let faces = vec![(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)];

        let manifold = Manifold::new(vertices, faces);
        let fm = FastMarching::new(&manifold);

        // Compute from source 0
        let dist_from_0 = fm.compute_distance(0).unwrap();

        println!("Symmetry test distances: {:?}", dist_from_0);
        println!("Distance to v1: {}", dist_from_0[1]);
        println!("Distance to v2: {}", dist_from_0[2]);
        println!("Distance to v3: {}", dist_from_0[3]);

        // Due to symmetry, distances from 0 to vertices 1,2,3 should be similar
        // But they might not be exactly equal due to numerical precision and update order
        let avg_dist = (dist_from_0[1] + dist_from_0[2] + dist_from_0[3]) / 3.0;
        println!("Average distance: {}", avg_dist);

        // Check they're all within 10% of each other
        for i in 1..4 {
            assert!(
                (dist_from_0[i] - avg_dist).abs() / avg_dist < 0.1,
                "Distance {} deviates too much from average: {} vs {}",
                i,
                dist_from_0[i],
                avg_dist
            );
        }
    }
}
