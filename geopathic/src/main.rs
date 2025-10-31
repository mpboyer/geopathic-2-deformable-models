use geopathic::colormaps::distance_colormap;
use geopathic::edp::HeatMethod;
use geopathic::loader::load_manifold;
use geopathic::viewer::Viewer;

fn main() {
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
    // viewer.draw_path(&path, None);
    viewer.render(true);
}
