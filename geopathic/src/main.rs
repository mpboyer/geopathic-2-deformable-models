// use geopathic::loader::load_manifold;
use geopathic::manifold::Manifold;
use geopathic::manifold::Point;
use geopathic::viewer::display_manifold;

fn main() {
    // let manifold = load_manifold("../examples/models/teapot.obj").unwrap();
    let manifold = Manifold::new(
        vec![
            (0, Point::from_row_slice(&[0.0, 0.0, 0.0])),
            (1, Point::from_row_slice(&[1.0, 0.0, 0.0])),
            (2, Point::from_row_slice(&[0.0, 1.0, 0.0])),
            (3, Point::from_row_slice(&[0.0, 0.0, 1.0])),
        ]
        .into_iter()
        .collect(),
        vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
    );
    display_manifold(&manifold);
}
