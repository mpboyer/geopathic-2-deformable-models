extern crate kiss3d;
extern crate nalgebra as na;

use kiss3d::light::Light;
use kiss3d::prelude::Mesh;
use kiss3d::window::Window;
use na::{UnitQuaternion, Vector3};
use nalgebra::Point3;

use std::cell::RefCell;
use std::rc::Rc;

use crate::manifold::Manifold;

/// Displays the given manifold using kiss3d
pub fn display_manifold(manifold: &Manifold) {
    let mut window = Window::new("Manifold Viewer");

    // add the coordinates of the manifold vertices (with respect to the order in the hashmap)
    let mut coords: Vec<Point3<f32>> = Vec::with_capacity(manifold.vertices.len());
    for i in 0..manifold.vertices.len() {
        let vertex = &manifold.vertices[i];
        coords.push(Point3::new(vertex[0], vertex[1], vertex[2]));
    }

    // add the faces
    let faces: Vec<Point3<u16>> = manifold
        .faces
        .iter() // only triangles for now
        .map(|f| Point3::new(f.0 as u16, f.1 as u16, f.2 as u16))
        .collect();

    // create a mesh and add it to the window
    let mesh = Mesh::new(coords, faces, None, None, true);
    let mesh = Rc::new(RefCell::new(mesh));
    let mut mesh_node = window.add_mesh(mesh, Vector3::new(1.0, 1.0, 1.0));

    // draw the edges
    mesh_node.set_color(1.0, 1.0, 1.0);
    mesh_node.set_lines_color(Some(Point3::new(1.0, 0.0, 0.0)));
    mesh_node.set_lines_width(1.0);
    mesh_node.enable_backface_culling(false);

    // set the lighting
    window.set_light(Light::StickToCamera);

    // rotate the object a bit each frame
    let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.005);
    while window.render() {
        mesh_node.prepend_to_local_rotation(&rot);
    }
}
