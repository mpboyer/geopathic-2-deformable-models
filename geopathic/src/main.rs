use std::io::Write;

use geopathic::colormaps::{distance_colormap, iso_distances};
use geopathic::edp::EDPMethod;
use geopathic::fastmarching::FastMarching;
use geopathic::ich::ICH;
use geopathic::loader::load_manifold;
use geopathic::mesh::Mesh;
use geopathic::viewer::Viewer;
use kiss3d::camera::ArcBall;
use nalgebra::{DVector, Point3};

fn main() {
    // heat_method();
    // fast_marching();
    ich();
}

#[allow(dead_code)]
fn heat_method() {
    let manifold = load_manifold("../examples/models/teddy.obj").unwrap();
    let heat_method = EDPMethod::new(&manifold, 0.5);

    let sources = [0, 431];
    let distances = match heat_method.compute_distance_heat(sources) {
        Ok(it) => it,
        Err(err) => panic!("{}", err),
    };
    let colormap = distance_colormap(&manifold, &distances);

    let mut viewer = Viewer::new();
    viewer.add_manifold(&manifold, Some(colormap));
    viewer.plot_curves(iso_distances(&manifold, &distances, 1e-6));

    for s in sources {
        viewer.draw_point(
            manifold.vertices()[s].clone(),
            Some(10.0),
            Some(Point3::from_slice(&[0.0, 0.0, 1.0])),
        );
    }

    viewer.camera = ArcBall::new(Point3::new(0.0, 10.0, 65.0), Point3::new(0.0, 0.0, 0.0));
    viewer.render(true);
}

#[allow(dead_code)]
fn fast_marching() {
    let manifold = load_manifold("../examples/models/teddy.obj").unwrap();
    let fastmarching = FastMarching::new(&manifold);
    let sources = [0, 256];
    let distances = match fastmarching.compute_distance(sources) {
        Ok(it) => it,
        Err(err) => panic!("{}", err),
    };
    let colormap = distance_colormap(&manifold, &distances);

    let mut viewer = Viewer::new();
    viewer.add_manifold(&manifold, Some(colormap));
    viewer.plot_curves(iso_distances(&manifold, &distances, 1.0));

    viewer.plot_sources(&manifold, sources, None, None);

    viewer.camera = ArcBall::new(Point3::new(0.0, 10.0, 65.0), Point3::new(0.0, 0.0, 0.0));
    viewer.render(true);
}

#[allow(dead_code)]
fn ich() {
    let manifold = load_manifold("../examples/models/rabbit-low-poly.obj").unwrap();
    print!("Converting to mesh...");
    std::io::stdout().flush().unwrap();
    let mesh = Mesh::from_manifold(&manifold);
    mesh.check_topology().unwrap();
    println!("done.");

    print!("Running ICH...");
    std::io::stdout().flush().unwrap();
    let mut ich = ICH::new(mesh, vec![50], vec![], vec![]);
    ich.run();
    println!("done.");
    ich.print_stats();

    let distances = ich.distances_to_vertices();
    let colormap = distance_colormap(&manifold, &DVector::from_vec(distances));

    let mut viewer = Viewer::new();
    viewer.white_background();
    viewer.add_manifold(&manifold, Some(colormap));

    for dest in 1..manifold.vertices().len() {
        let path: Vec<DVector<f64>> = ich
            .path_to_vertex(dest)
            .iter()
            .map(|p| DVector::from_vec(vec![p.x, p.y, p.z]))
            .collect();
        viewer.draw_path(&path, Some(20.0), None);
    }

    println!("Rendering...");
    viewer.render(false);
}
