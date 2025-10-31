use nalgebra::DVector;

use crate::manifold::Manifold;

pub fn random_colormap(manifold: &Manifold) -> Vec<[u8; 4]> {
    (0..manifold.faces().len())
        .map(|i| {
            let r = ((i * 37) % 256) as u8;
            let g = ((i * 59) % 256) as u8;
            let b = ((i * 83) % 256) as u8;
            [r, g, b, 255]
        })
        .collect()
}

/// Create a colormap with a gradient from blue to red depending on the height of the face centroid
pub fn vertical_colormap(manifold: &Manifold) -> Vec<[u8; 4]> {
    let mut colormap = Vec::with_capacity(manifold.faces().len());
    // compute the lowest and highest y among vertices
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for vertex in manifold.vertices() {
        if vertex[1] < min_y {
            min_y = vertex[1];
        }
        if vertex[1] > max_y {
            max_y = vertex[1];
        }
    }
    for face in manifold.faces() {
        let v0 = &manifold.vertices()[face.0];
        let v1 = &manifold.vertices()[face.1];
        let v2 = &manifold.vertices()[face.2];
        let centroid_y = (v0[1] + v1[1] + v2[1]) / 3.0;
        let t = (centroid_y - min_y) / (max_y - min_y);
        let r = (t * 255.0) as u8;
        let g = 0;
        let b = ((1.0 - t) * 255.0) as u8;
        colormap.push([r, g, b, 255]);
    }
    colormap
}

/// Create a colormap from distance values, mapping from purple (#7d1dd3) to yellow (#ffe500)
/// The distance value for each face is the mean of its vertex distances
pub fn distance_colormap(manifold: &Manifold, distances: &DVector<f32>) -> Vec<[u8; 4]> {
    let mut colormap = Vec::with_capacity(manifold.faces().len());

    // Find min and max distances for normalization
    let min_dist = distances.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_dist = distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_dist - min_dist;

    // Color endpoints: purple #7d1dd3 to yellow #ffe500
    let color_start = [0x7d as f32, 0x1d as f32, 0xd3 as f32]; // Purple
    let color_end = [0xff as f32, 0xe5 as f32, 0x00 as f32]; // Yellow

    for face in manifold.faces() {
        let (i, j, k) = *face;

        // Compute mean distance for this face
        let mean_dist = (distances[i] + distances[j] + distances[k]) / 3.0;

        // Normalize to [0, 1]
        let t = if range > 1e-10 {
            (mean_dist - min_dist) / range
        } else {
            0.0
        };

        // Interpolate colors
        let r = (color_start[0] + t * (color_end[0] - color_start[0])) as u8;
        let g = (color_start[1] + t * (color_end[1] - color_start[1])) as u8;
        let b = (color_start[2] + t * (color_end[2] - color_start[2])) as u8;

        colormap.push([r, g, b, 255]);
    }

    colormap
}
