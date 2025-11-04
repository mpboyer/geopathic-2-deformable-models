use std::io::Write;

use geopathic::colormaps::{distance_colormap, iso_distances};
use geopathic::edp::HeatMethod;
use geopathic::ich::ICH;
use geopathic::loader::load_manifold;
// use geopathic::manifold::Path;
use geopathic::mesh::Mesh;
use geopathic::viewer::Viewer;
use kiss3d::camera::ArcBall;
use nalgebra::{DVector, Point3};

fn main() {
    // heat_method();
    ich();
}

#[allow(dead_code)]
fn heat_method() {
    let manifold = load_manifold("../examples/models/teddy.obj").unwrap();
    let heat_method = HeatMethod::new(&manifold, 10.0);
    let distances = match heat_method.compute_distance(0) {
        Ok(it) => it,
        Err(err) => panic!("{}", err),
    };
    let colormap = distance_colormap(&manifold, &distances);
    println!("{}", distances);

    let mut viewer = Viewer::new();
    viewer.add_manifold(&manifold, Some(colormap));
    viewer.plot_curves(iso_distances(&manifold, &distances, 1e-6));
    viewer.render(true);
}

#[allow(dead_code)]
fn ich() {
    let manifold = load_manifold("../examples/models/teddy.obj").unwrap();
    print!("Converting to mesh...");
    std::io::stdout().flush().unwrap();
    let mesh = Mesh::from_manifold(&manifold);
    println!("done.");

    print!("Running ICH...");
    std::io::stdout().flush().unwrap();
    let mut ich = ICH::new(mesh, vec![0], vec![], vec![]);
    ich.run();
    println!("done.");
    ich.print_stats();

    let distances = ich
        .distances_to_vertices()
        .iter()
        .map(|&d| d as f32)
        .collect();
    let colormap = distance_colormap(&manifold, &DVector::from_vec(distances));

    // let path: Path = ich.path_to_vertex(100).iter().map(|p| DVector::from_vec(vec![p.x as f32, p.y as f32, p.z as f32])).collect();

    let mut viewer = Viewer::new();
    viewer.add_manifold(&manifold, Some(colormap));
    viewer.camera = ArcBall::new(Point3::new(0.0, 10.0, 65.0), Point3::new(0.0, 0.0, 0.0));
    // viewer.draw_path(&path, Some(5.0), None);
    viewer.render(false);
}
