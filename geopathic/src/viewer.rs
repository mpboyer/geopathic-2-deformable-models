extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::prelude::Mesh;
use kiss3d::window::Window;
use kiss3d::{light::Light, prelude::TextureManager};
use na::{Point2, UnitQuaternion, Vector3};
use nalgebra::Point3;

use std::cell::RefCell;
use std::rc::Rc;

use image::{DynamicImage, GenericImage};

use crate::manifold::Manifold;

/// Displays the given manifold using kiss3d
pub fn display_manifold(manifold: &Manifold, face_colormap: Option<Vec<[u8; 4]>>) {
    let mut window = Window::new("Manifold Viewer");

    // add the coordinates of the manifold vertices (with respect to the order in the hashmap)
    let mut base_coords: Vec<Point3<f32>> = Vec::with_capacity(manifold.vertices.len());
    for i in 0..manifold.vertices.len() {
        let vertex = &manifold.vertices[i];
        base_coords.push(Point3::new(vertex[0], vertex[1], vertex[2]));
    }

    // add the faces
    let base_faces: Vec<Point3<u16>> = manifold
        .faces
        .iter()
        .map(|f| Point3::new(f.0 as u16, f.1 as u16, f.2 as u16))
        .collect();

    let (mesh, texture) = match face_colormap {
        Some(colormap) => {
            // create a 1xN texture where each texel = face color
            let face_count = base_faces.len() as u32;
            let mut image = DynamicImage::new_rgba8(face_count, 1);
            for i in 0..face_count {
                image.put_pixel(i, 0, image::Rgba(colormap[i as usize]));
            }

            // duplicate vertices per face and assign UVs pointing to the right texel
            let mut coords = Vec::new();
            let mut uvs = Vec::new();
            let mut faces = Vec::new();

            for (i, f) in base_faces.iter().enumerate() {
                let base_u = (i as f32 + 0.5) / (face_count as f32); // center of texel
                for j in 0..3 {
                    coords.push(base_coords[f[j] as usize]);
                    uvs.push(Point2::new(base_u, 0.5)); // same UV for all 3 vertices
                }
                let idx = (i * 3) as u16;
                faces.push(Point3::new(idx, idx + 1, idx + 2));
            }

            let texture =
                TextureManager::get_global_manager(|tm| tm.add_image(image.clone(), "colors"));

            (
                Mesh::new(coords, faces, None, Some(uvs), true),
                Some(texture),
            )
        }
        None => (Mesh::new(base_coords, base_faces, None, None, true), None),
    };

    // create a mesh and add it to the window
    let mesh = Rc::new(RefCell::new(mesh));
    let mut mesh_node = window.add_mesh(mesh, Vector3::new(1.0, 1.0, 1.0));
    mesh_node.enable_backface_culling(false);

    // add the texture (color per face) or set a default color
    match texture {
        Some(tex) => {
            mesh_node.set_texture(tex);
        }
        None => {
            mesh_node.set_color(1.0, 1.0, 1.0);
            // draw the edges
            mesh_node.set_lines_color(Some(Point3::new(1.0, 0.0, 0.0)));
            mesh_node.set_lines_width(1.0);
        }
    }

    // set the lighting
    window.set_light(Light::StickToCamera);

    // rotate the object a bit each frame
    let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.005);
    while window.render() {
        mesh_node.prepend_to_local_rotation(&rot);
    }
}
