use geopathic::colormaps::vertical_colormap;
use geopathic::loader::load_manifold;
use geopathic::viewer::Viewer;

fn main() {
    let manifold = load_manifold("../examples/models/teapot.obj").unwrap();
    let colormap = vertical_colormap(&manifold);

    let mut viewer = Viewer::new();
    viewer.add_manifold(&manifold, Some(colormap));
    viewer.render();
}
