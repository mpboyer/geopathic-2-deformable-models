use geopathic::colormaps::vertical_colormap;
use geopathic::loader::load_manifold;
use geopathic::viewer::Viewer;

fn main() {
    let manifold = load_manifold("../examples/models/teapot.obj").unwrap();
    let colormap = vertical_colormap(&manifold);
    let face = manifold.faces()[100];
    let path = vec![
        manifold.vertices()[face.0].clone(),
        manifold.vertices()[face.1].clone(),
        manifold.vertices()[face.2].clone(),
    ];

    let mut viewer = Viewer::new();
    viewer.add_manifold(&manifold, Some(colormap));
    viewer.draw_path(&path, None);
    viewer.render(false);
}
