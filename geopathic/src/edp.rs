use core::f32;
use std::collections::HashMap;

use crate::manifold::{Manifold, Point};
use nalgebra::{DMatrix, DVector};

fn compute_cotangent(a: &Point, b: &Point) -> f32 {
    let dot = a.dot(b);
    let cross = a.cross(b);
    let cross_norm = cross.norm();

    if cross_norm > 1e-10 {
        dot / cross_norm
    } else {
        0.0
    }
}

impl Manifold {
    fn compute_face_gradient(
        &self,
        face_idx: usize,
        vertex_values: &DVector<f32>,
    ) -> Result<DVector<f32>, String> {
        let (i, j, k) = self.faces[face_idx];

        let v0 = &self.vertices[i];
        let v1 = &self.vertices[j];
        let v2 = &self.vertices[k];

        let u0 = vertex_values[i];
        let u1 = vertex_values[j];
        let u2 = vertex_values[k];

        let e0 = v2 - v1;
        let e1 = v0 - v2;
        let e2 = v1 - v0;

        // Compute face normal
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(&edge2);
        let double_area = normal.norm();

        if double_area < 1e-10 {
            return Err("Degenerate face encountered".to_string());
        }

        let n = normal / double_area;

        // Gradient using barycentric formula

        let grad =
            ((&n.cross(&e0) * u0) + (&n.cross(&e1) * u1) + (&n.cross(&e2) * u2)) / double_area;

        Ok(grad)
    }

    fn compute_divergence(&self, face_vectors: &[DVector<f32>]) -> Result<DVector<f32>, String> {
        let n = self.vertices.len();
        let mut divergence = DVector::zeros(n);

        for (face_idx, face) in self.faces.iter().enumerate() {
            let (i, j, k) = *face;

            let vi = &self.vertices[i];
            let vj = &self.vertices[j];
            let vk = &self.vertices[k];

            let x = &face_vectors[face_idx];

            // Edge vectors
            let eij = vj - vi;
            let eik = vk - vi;
            let ejk = vk - vj;

            // Cotangent at each vertex
            let cot_k = compute_cotangent(&eik, &(-&ejk));
            let cot_i = compute_cotangent(&(-&eij), &eik);
            let cot_j = compute_cotangent(&eij, &ejk);

            // Add contributions to divergence
            divergence[i] += 0.5 * (cot_j * (eij.dot(x)) + cot_k * (eik.dot(x)));
            divergence[j] += 0.5 * (cot_i * ((-&eij).dot(x)) + cot_k * (ejk.dot(x)));
            divergence[k] += 0.5 * (cot_i * ((-&eik).dot(x)) + cot_j * ((-&ejk).dot(x)));
        }

        Ok(divergence)
    }
}

#[derive(Debug, Clone)]
pub struct Laplacian {
    pub laplace_matrix: DMatrix<f32>,
    pub n_vertices: usize,
}

impl Laplacian {
    pub fn new(manifold: &Manifold) -> Self {
        let n_vertices = manifold.vertices().len();
        let mut laplace_matrix = DMatrix::zeros(n_vertices, n_vertices);

        let mut edge_weights: HashMap<(usize, usize), f32> = HashMap::new();

        for face in manifold.faces() {
            let (i, j, k) = *face;

            let vi = &manifold.vertices()[i];
            let vj = &manifold.vertices()[j];
            let vk = &manifold.vertices()[k];

            // Compute cotangent weights for each edge
            Self::add_cotangent_weight(&mut edge_weights, i, j, k, vi, vj, vk);
            Self::add_cotangent_weight(&mut edge_weights, j, k, i, vj, vk, vi);
            Self::add_cotangent_weight(&mut edge_weights, k, i, j, vk, vi, vj);
        }

        for ((i, j), weight) in edge_weights.iter() {
            laplace_matrix[(*i, *j)] = -weight;
            laplace_matrix[(*j, *i)] = -weight;
        }

        // Set diagonal entries (sum of incident edge weights)
        for i in 0..n_vertices {
            let row_sum: f32 = (0..n_vertices)
                .filter(|&j| j != i)
                .map(|j| -laplace_matrix[(i, j)])
                .sum();
            laplace_matrix[(i, i)] = row_sum;
        }

        Self {
            laplace_matrix,
            n_vertices,
        }
    }

    fn add_cotangent_weight(
        edge_weights: &mut HashMap<(usize, usize), f32>,
        i: usize,
        j: usize,
        _k: usize,
        vi: &Point,
        vj: &Point,
        vk: &Point,
    ) {
        let ki = vi - vk;
        let kj = vj - vk;

        let dot = ki.dot(&kj);
        let cross_norm = ki.cross(&kj).norm();

        if cross_norm > 1e-10 {
            let cot = dot / cross_norm;
            let weight = 0.5 * cot;

            *edge_weights.entry((i.min(j), i.max(j))).or_insert(0.0) += weight;
        }
    }

    pub fn apply(&self, u: &DVector<f32>) -> DVector<f32> {
        &self.laplace_matrix * u
    }

    pub fn matrix(&self) -> &DMatrix<f32> {
        &self.laplace_matrix
    }
}

pub struct HeatMethod<'a> {
    manifold: &'a Manifold,
    pub laplace: Laplacian,
    time_step: f32,
}

impl<'a> HeatMethod<'a> {
    pub fn new(manifold: &'a Manifold, time_step: f32) -> Self {
        Self {
            manifold,
            laplace: Laplacian::new(manifold),
            time_step,
        }
    }

    pub fn compute_distance(&self, source: usize) -> Result<DVector<f32>, String> {
        let n = self.manifold.vertices().len();

        // (I + t*L)u = δ_source
        let identity = DMatrix::identity(n, n);
        let heat_matrix = &identity + &self.laplace.laplace_matrix * self.time_step;

        let mut rhs = DVector::zeros(n);
        rhs[source] = 1.0;

        let u = Self::solve_linear_system(&heat_matrix, &rhs)?;
        // println!("U: {}", u.clone());

        // X = -∇u / |∇u|
        let mut face_gradients = Vec::new();

        for face_idx in 0..self.manifold.faces().len() {
            let gradient = self.manifold.compute_face_gradient(face_idx, &u)?;

            let grad_norm = gradient.norm();
            let normalized_grad = if grad_norm > 1e-6 {
                -gradient / grad_norm
            } else {
                DVector::zeros(3)
            };
            face_gradients.push(normalized_grad);
        }

        // Solve Poisson equation L*φ = ∇·X
        let divergence_x = self.manifold.compute_divergence(&face_gradients)?;

        let regularization = 1e-2;
        let regularized_laplacian = &self.laplace.laplace_matrix + &identity * regularization;

        let phi = Self::solve_linear_system(&regularized_laplacian, &divergence_x)?;
        let distances = phi.map(|x| x - phi[source]);

        Ok(distances)
    }

    fn solve_linear_system(a: &DMatrix<f32>, b: &DVector<f32>) -> Result<DVector<f32>, String> {
        match a.clone().cholesky().ok_or_else(|| {
            "Failed to solve linear system (matrix may be singular or ill-conditioned)".to_string()
        }) {
            Ok(ll) => Ok(ll.solve(b)),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_construction() {
        let vertices = [
            DVector::from_vec(vec![0.0, 0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0]),
            DVector::from_vec(vec![0.0, 0.0, 1.0]),
        ];

        let faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)];

        let manifold = Manifold::new(vertices.to_vec(), faces.to_vec());
        let laplace = Laplacian::new(&manifold);

        assert_eq!(laplace.matrix().nrows(), 4);
        assert_eq!(laplace.matrix().ncols(), 4);

        for i in 0..4 {
            let row_sum: f32 = (0..4).map(|j| laplace.matrix()[(i, j)]).sum();
            assert!(row_sum.abs() < 1e-5, "Row {} sum: {}", i, row_sum);
        }
    }
}
