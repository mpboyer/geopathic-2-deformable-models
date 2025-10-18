use geopathic::loader::load_manifold;
use geopathic::viewer::display_manifold;

fn main() {
    let manifold = load_manifold("../examples/models/pyramid.obj").unwrap();
    display_manifold(&manifold);
}
