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
