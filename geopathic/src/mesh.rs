//! Defines a Mesh structure, extending Manifold by precomputing some data.

use nalgebra::Point3;

use crate::manifold;

/// Vertex of a mesh
#[derive(Debug, Clone, PartialEq)]
pub struct Vertex {
    /// Position of the vertex
    pub(crate) position: Point3<f64>,
    /// List of the indices of adjacent edges
    pub(crate) edges: Vec<usize>,
}

/// Edge of a mesh
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// Index of the start vertex
    pub(crate) start: usize,
    /// Index of the end vertex
    pub(crate) end: usize,
    /// Index of the next edge around the face
    pub(crate) next_edge: usize,
    /// Index of the opposite twin edge
    pub(crate) twin_edge: Option<usize>,
    /// Length of the edge
    pub(crate) length: f64,
}

/// Mesh, containing vertices, edge and faces
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) edges: Vec<Edge>,
    /// List of faces, each defined by a triplet of vertex indices
    pub(crate) faces: Vec<[usize; 3]>,
    /// Precomputed sum of angles of each vertex
    pub(crate) angles: Vec<f64>,
}

impl Mesh {
    /// Create a Mesh from a Manifold, precomputing necessary data.
    pub fn from_manifold(manifold: &manifold::Manifold) -> Self {
        let mut vertices = Vec::new();
        let mut edges: Vec<Edge> = Vec::new();
        let mut faces = Vec::new();
        let mut angles = vec![0.0; manifold.faces().len()];

        // Iterate over the vertices in the manifold
        for v in &manifold.vertices {
            let position = Point3::new(v[0] as f64, v[1] as f64, v[2] as f64);
            vertices.push(Vertex {
                position,
                edges: Vec::new(),
            });
        }

        // Iterate over the faces in the manifold
        for f in &manifold.faces {
            faces.push([f.0, f.1, f.2]);
        }

        // Compute the edges
        for face in &faces {
            let v0 = &vertices[face[0]].position;
            let v1 = &vertices[face[1]].position;
            let v2 = &vertices[face[2]].position;

            let edge_lengths = [(v1 - v0).norm(), (v2 - v1).norm(), (v0 - v2).norm()];

            // Create edges
            let edge_indices = [edges.len(), edges.len() + 1, edges.len() + 2];

            for i in 0..3 {
                let start = face[i];
                let end = face[(i + 1) % 3];
                let next_edge = edge_indices[(i + 1) % 3];

                edges.push(Edge {
                    start,
                    end,
                    next_edge,
                    twin_edge: None,
                    length: edge_lengths[i],
                });

                vertices[start].edges.push(edge_indices[i]);
            }
        }

        // Compute the twin edges
        for i in 0..edges.len() {
            for j in 0..edges.len() {
                if edges[i].start == edges[j].end && edges[i].end == edges[j].start {
                    edges[i].twin_edge = Some(j);
                }
            }
        }

        // Compute angles at each vertex (see code above)
        for (i, face) in faces.iter().enumerate() {
            for j in 0..3 {
                let cur_vert = face[j];

                // TODO: precompute this in the faces

                // Find the edge from cur_vert to the next vertex in the face
                let edge0_idx = vertices[cur_vert]
                    .edges
                    .iter()
                    .find(|&&e_idx| edges[e_idx].end == face[(j + 1) % 3])
                    .copied()
                    .expect("Edge not found");
                let edge0 = edges[edge0_idx].length;

                // Find the edge from cur_vert to the previous vertex in the face
                let edge1_idx = vertices[cur_vert]
                    .edges
                    .iter()
                    .find(|&&e_idx| edges[e_idx].end == face[(j + 2) % 3])
                    .copied()
                    .expect("Edge not found");
                let edge1 = edges[edge1_idx].length;

                let vec0 = vertices[cur_vert].position;
                let vec1 = vertices[face[(j + 1) % 3]].position;
                let vec2 = vertices[face[(j + 2) % 3]].position;

                let edge2 = (vec1 - vec2).norm();

                let cur_angle = ((edge0.powi(2) + edge1.powi(2) - edge2.powi(2))
                    / (2.0 * edge0 * edge1))
                    .acos();
                // TODO: handle concave angles properly if necessary
                angles[cur_vert] += cur_angle;
                if i == 1 && j == 2 {
                    // print everything
                    eprintln!("Face {} Vertex {} ({:?})", i, j, face);
                    eprintln!("Vertex indices: {}, {}, {}", face[j], face[(j+1) % 3], face[(j+2) % 3]);
                    eprintln!(
                        "edge0: {}, edge1: {}, edge2: {}",
                        edge0, edge1, edge2
                    );
                    eprintln!(
                        "vec0: ({}, {}, {}), vec1: ({}, {}, {}), vec2: ({}, {}, {})",
                        vec0.x, vec0.y, vec0.z, vec1.x, vec1.y, vec1.z, vec2.x, vec2.y, vec2.z
                    );
                    // eprintln!("i: {}, j: {}, angle: {}", i, j, cur_angle);
                }
            }
        }

        Mesh {
            vertices,
            edges,
            faces,
            angles,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::load_manifold;
    use approx::assert_abs_diff_eq;

    #[test]
    fn manifold_to_mesh_pyramid() {
        let manifold = load_manifold("../examples/models/pyramid.obj").unwrap();
        let mesh = Mesh::from_manifold(&manifold);

        let expected_mesh = Mesh {
            vertices: vec![
                Vertex {
                    position: Point3::new(0.0, 0.0, 0.0),
                    edges: vec![0, 3, 6],
                },
                Vertex {
                    position: Point3::new(1.0, 0.0, 0.0),
                    edges: vec![1, 8, 9],
                },
                Vertex {
                    position: Point3::new(0.0, 1.0, 0.0),
                    edges: vec![2, 4, 11],
                },
                Vertex {
                    position: Point3::new(0.0, 0.0, 1.0),
                    edges: vec![5, 7, 10],
                },
            ],
            edges: vec![
                Edge {
                    start: 0,
                    end: 1,
                    next_edge: 1,
                    twin_edge: Some(8),
                    length: 1.0,
                },
                Edge {
                    start: 1,
                    end: 2,
                    next_edge: 2,
                    twin_edge: Some(11),
                    length: f64::sqrt(2.0),
                },
                Edge {
                    start: 2,
                    end: 0,
                    next_edge: 0,
                    twin_edge: Some(3),
                    length: 1.0,
                },
                Edge {
                    start: 0,
                    end: 2,
                    next_edge: 4,
                    twin_edge: Some(2),
                    length: 1.0,
                },
                Edge {
                    start: 2,
                    end: 3,
                    next_edge: 5,
                    twin_edge: Some(10),
                    length: f64::sqrt(2.0),
                },
                Edge {
                    start: 3,
                    end: 0,
                    next_edge: 3,
                    twin_edge: Some(6),
                    length: 1.0,
                },
                Edge {
                    start: 0,
                    end: 3,
                    next_edge: 7,
                    twin_edge: Some(5),
                    length: 1.0,
                },
                Edge {
                    start: 3,
                    end: 1,
                    next_edge: 8,
                    twin_edge: Some(9),
                    length: f64::sqrt(2.0),
                },
                Edge {
                    start: 1,
                    end: 0,
                    next_edge: 6,
                    twin_edge: Some(0),
                    length: 1.0,
                },
                Edge {
                    start: 1,
                    end: 3,
                    next_edge: 10,
                    twin_edge: Some(7),
                    length: f64::sqrt(2.0),
                },
                Edge {
                    start: 3,
                    end: 2,
                    next_edge: 11,
                    twin_edge: Some(4),
                    length: f64::sqrt(2.0),
                },
                Edge {
                    start: 2,
                    end: 1,
                    next_edge: 9,
                    twin_edge: Some(1),
                    length: f64::sqrt(2.0),
                },
            ],
            faces: vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
            angles: vec![4.71238898, 2.617993878, 2.617993878, 2.617993878],
        };

        assert_eq!(mesh.vertices, expected_mesh.vertices);
        assert_eq!(mesh.edges, expected_mesh.edges);
        assert_eq!(mesh.faces, expected_mesh.faces);
        for (a, b) in mesh.angles.iter().zip(expected_mesh.angles.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn manifold_to_mesh_teddy() {
        let manifold = load_manifold("../examples/models/teddy.obj").unwrap();
        let _ = Mesh::from_manifold(&manifold);
    }
}
