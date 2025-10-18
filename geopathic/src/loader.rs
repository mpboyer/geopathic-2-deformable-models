use std::collections::HashMap;

use crate::manifold::Manifold;
use crate::manifold::Point;

use obj::Obj;

pub fn load_manifold(file_path: &str) -> Result<Manifold, obj::ObjError> {
    let object = Obj::load(file_path)?;

    let mut vertices: HashMap<usize, Point> = HashMap::new();
    let mut faces = Vec::new();

    for (i, vertex) in object.data.position.iter().enumerate() {
        vertices.insert(i, Point::from_row_slice(vertex));
    }

    for sub_obj in object.data.objects {
        for group in sub_obj.groups {
            for poly in group.polys {
                let indices: Vec<usize> = poly.0.iter().map(|v| v.0).collect();
                faces.push(indices);
            }
        }
    }

    Ok(Manifold::new(vertices, faces))
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_load_manifold_teapot() {
        let result = load_manifold("../examples/models/teapot.obj");
        let manifold = result.unwrap();
        dbg!(manifold.vertices.len());
        dbg!(manifold.faces.len());
    }

    #[test]
    fn test_load_manifold_cow() {
        let result = load_manifold("../examples/models/cow-nonormals.obj");
        let manifold = result.unwrap();
        dbg!(manifold.vertices.len());
        dbg!(manifold.faces.len());
    }

    #[test]
    fn test_load_manifold_teddy() {
        let result = load_manifold("../examples/models/teddy.obj");
        let manifold = result.unwrap();
        dbg!(manifold.vertices.len());
        dbg!(manifold.faces.len());
    }
}
