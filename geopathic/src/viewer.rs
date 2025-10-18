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

pub fn display_manifold(manifold: &Manifold) {
    let mut window = Window::new("Manifold Viewer");

    // add the manifold to the window
    let coords: Vec<Point3<f32>> = manifold
        .vertices
        .values()
        .map(|p| Point3::new(p[0], p[1], p[2]))
        .collect();
    let faces: Vec<Point3<u16>> = manifold
        .faces
        .iter()
        .filter(|&f| f.len() == 3) // only triangles for now
        .map(|f| Point3::new(f[0] as u16, f[1] as u16, f[2] as u16))
        .collect();

    let mesh = Mesh::new(coords, faces, None, None, true);
    let mesh = Rc::new(RefCell::new(mesh));
    let mut mesh_node = window.add_mesh(mesh, Vector3::new(1.0, 1.0, 1.0));

    // draw the edges
    mesh_node.set_color(1.0, 1.0, 1.0);
    mesh_node.set_lines_color(Some(Point3::new(1.0, 0.0, 0.0)));
    mesh_node.set_lines_width(100_000.0);
    mesh_node.enable_backface_culling(false);

    // set the light
    window.set_light(Light::StickToCamera);

    // rotate the object a bit each frame
    let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.005);
    while window.render() {
        mesh_node.prepend_to_local_rotation(&rot);
    }
}
