use std::io::Write;

use geopathic::edp::HeatMethod;
use geopathic::ich::ICH;
use geopathic::loader::load_manifold;
use geopathic::viewer::Viewer;
use geopathic::{colormaps::distance_colormap, mesh::Mesh};
// use kiss3d::camera::ArcBall;
// use nalgebra::{DVector, Point3};

fn main() {
    heat_method();
    // ich();
}

#[allow(dead_code)]
fn heat_method() {
    let manifold = load_manifold("../examples/models/teapot.obj").unwrap();
    let heat_method = HeatMethod::new(&manifold, 1e-6);
    let distances = match heat_method.compute_distance(0) {
        Ok(it) => it,
        Err(err) => panic!("{}", err),
    };
    let colormap = distance_colormap(&manifold, &distances);
    println!("{}", distances);

    let mut viewer = Viewer::new();
    viewer.add_manifold(&manifold, Some(colormap));
    viewer.render(true);
}

#[allow(dead_code)]
fn ich() {
    let manifold = load_manifold("../examples/models/pyramid.obj").unwrap();
    print!("Converting to mesh...");
    std::io::stdout().flush().unwrap();
    let mesh = Mesh::from_manifold(&manifold);
    println!("done.");
    print!("Running ICH...");
    std::io::stdout().flush().unwrap();
    let mut ich = ICH::new(mesh, vec![1], vec![], vec![]);
    ich.run();
    println!("done.");
    ich.print_stats();

    // // path between vertices 5 and 100
    // let path = vec![
    //     DVector::from_column_slice(&[1.22297, 8.23418, 8.54689]),
    //     DVector::from_column_slice(&[0.737699, 9.56941, 8.38842]),
    //     DVector::from_column_slice(&[0.123089, 11.2351, 8.13982]),
    //     DVector::from_column_slice(&[0.0186267, 11.5161, 8.09203]),
    //     DVector::from_column_slice(&[-0.760558, 13.3092, 7.08088]),
    //     DVector::from_column_slice(&[-1.12477, 14.1202, 6.5516]),
    //     DVector::from_column_slice(&[-1.31237, 14.4422, 6.10272]),
    //     DVector::from_column_slice(&[-2.09717, 15.7819, 4.21627]),
    //     DVector::from_column_slice(&[-2.46202, 16.014, 3.04381]),
    //     DVector::from_column_slice(&[-2.58306, 16.091, 2.65485]),
    //     DVector::from_column_slice(&[-2.94553, 16.2995, 1.58103]),
    //     DVector::from_column_slice(&[-3.23342, 16.1334, 0.968515]),
    //     DVector::from_column_slice(&[-4.14648, 15.5999, -0.973337]),
    // ];

    let mut viewer = Viewer::new();
    viewer.add_manifold(&manifold, None);
    // viewer.draw_path(&path, Some(5.0), None);
    // viewer.draw_path(
    //     &vec![
    //         manifold.vertices()[5].clone(),
    //         manifold.vertices()[100].clone(),
    //     ],
    //     Some(5.0),
    //     None,
    // );
    // viewer.camera = ArcBall::new(Point3::new(0.0, 10.0, 65.0), Point3::new(0.0, 0.0, 0.0));
    viewer.render(false);
}
