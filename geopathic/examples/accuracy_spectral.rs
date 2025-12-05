use geopathic::edp::*;
use geopathic::loader::load_manifold;
use geopathic::sources::Sources;
use rand::Rng;
use std::fs::File;
use std::io::Write;

const METHODS: [SpectralPDE; 4] = [
    SpectralPDE::Eigenmap,
    SpectralPDE::CommuteTime,
    SpectralPDE::Diffusion,
    SpectralPDE::Biharmonic,
];

fn main() {
    // List all files in ../examples/models/
    let model_paths: Vec<_> = std::fs::read_dir("../examples/models/")
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            if path.extension()?.to_str()? == "obj" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    let skipped = [
        // non-manifold meshes
        "nonmanifold_edge.obj",
        "suzanne.obj",
        "teapot.obj",
        // weird propagation
        "cow-nonormals.obj",
        // too big
        // "xwings.obj",
        "screw.obj",
        "dragon.obj",
        "rabbit-head.obj",
        "fantasy-piece.obj",
        "ferrari.obj",
    ];
    let model_paths: Vec<_> = model_paths
        .into_iter()
        .filter(|path| !skipped.contains(&path.file_name().unwrap().to_str().unwrap()))
        .collect();

    let mut rng = rand::rng();
    let timestamp = chrono::Local::now().format("%m:%d_%H:%M:%S").to_string();
    let mut file = File::create(format!(
        "../benchmarks/csv/accuracy_spectral_methods_{}.csv",
        timestamp
    ))
    .unwrap();
    writeln!(file, "model,vertices,faces,method,dimension,source,time").unwrap();

    // warmup run
    println!("Warming up...\n");
    for _ in 0..10 {
        let manifold = load_manifold("../examples/models/teddy.obj").unwrap();
        let source = Sources::from(0);
        let edp = EDPMethod::new(&manifold);
        let spectral_method = SpectralPDE::CommuteTime;
        let k = 7;
        let _ = edp.compute_distance_spectral(source, spectral_method, k);
    }

    for path in model_paths {
        print!("Computing {:?}", path.file_name().unwrap());
        std::io::stdout().flush().unwrap();
        // load the manifold
        let manifold = load_manifold(path.to_str().unwrap());
        if manifold.is_err() {
            println!(
                "skipping {:?} due to load failure.",
                path.file_name().unwrap()
            );
            continue;
        }
        let manifold = manifold.unwrap();
        let n = manifold.vertices().len();

        for k in 0..n {
            for spectral_method in METHODS {
                let source = rng.random_range(0..manifold.vertices().len());

                let tick = std::time::Instant::now();
                let edp = EDPMethod::new(&manifold);
                let _ = edp.compute_distance_spectral(source, spectral_method, k);
                let time_elapsed = tick.elapsed();

                // log the results to csv
                writeln!(
                    file,
                    "{},{},{},{},{},{},{}",
                    path.file_name().unwrap().to_str().unwrap(),
                    manifold.vertices().len(),
                    manifold.faces().len(),
                    spectral_method,
                    k / n,
                    source,
                    time_elapsed.as_secs_f64(),
                )
                .unwrap();
                print!(".");
                std::io::stdout().flush().unwrap();
            }
        }
        println!("done.");
    }
}
