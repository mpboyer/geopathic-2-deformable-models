use geopathic::colormaps::vertical_colormap;
use geopathic::loader::load_manifold;
use geopathic::viewer::display_manifold;

fn main() {
    let manifold = load_manifold("../examples/models/teapot.obj").unwrap();
    let colormap = vertical_colormap(&manifold);

    display_manifold(&manifold, Some(colormap));
}
