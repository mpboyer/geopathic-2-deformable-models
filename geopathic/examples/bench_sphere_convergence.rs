use geopathic::fastmarching::*;
use geopathic::jetmarching::{
    InterpolantRepresentation, JetMarching, MinimizationProblemMethod, SlownessModel,
    StencilUpdateMethod,
};
use geopathic::loader::load_manifold;
use geopathic::manifold::Manifold;
use geopathic::sources::Sources;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use nalgebra::DVector;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

const SPHERE_FILES: &[&str] = &[
    // "sphere_12.obj",
    "sphere_42.obj",
    "sphere_162.obj",
    "sphere_642.obj",
    "sphere_2562.obj",
    "sphere_10242.obj",
    "sphere_40962.obj",
    "sphere_163842.obj",
    // "sphere_655362.obj",
];

const SPHERE_DIR: &str = "../examples/spheres/";

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const MAGENTA: &str = "\x1b[35m";
const RED: &str = "\x1b[31m";
const RESET: &str = "\x1b[0m";

fn c(colour: &str, text: &str) -> String {
    format!("{colour}{text}{RESET}")
}

struct UnitSlowness;
impl SlownessModel for UnitSlowness {
    fn at_vertex(&self, _idx: usize, _m: &Manifold) -> f64 {
        1.0
    }
    fn at_point_local(
        &self,
        _p: &DVector<f64>,
        _m: &Manifold,
        _l: &[usize],
        _v: &[Vec<usize>],
    ) -> f64 {
        1.0
    }
}

fn ground_truth(manifold: &Manifold, source: usize) -> Vec<f64> {
    let verts = manifold.vertices();
    let src_z = verts[source][2];
    debug_assert!(
        src_z.abs() < 1e-6,
        "Source vertex z = {src_z:.2e} — is the north pole really at the origin?"
    );

    verts
        .iter()
        .map(|v| {
            let cos_theta = (v[2] + 1.0).clamp(-1.0, 1.0);
            cos_theta.acos()
        })
        .collect()
}

fn mae(reference: &[f64], candidate: &[f64]) -> f64 {
    let (sum, count) = reference
        .iter()
        .zip(candidate)
        .filter(|(r, c)| r.is_finite() && c.is_finite())
        .fold((0.0f64, 0usize), |(s, n), (&r, &c)| {
            (s + (r - c).abs(), n + 1)
        });
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn max_abs_err(reference: &[f64], candidate: &[f64]) -> f64 {
    reference
        .iter()
        .zip(candidate)
        .filter(|(r, c)| r.is_finite() && c.is_finite())
        .map(|(&r, &c)| (r - c).abs())
        .fold(f64::NEG_INFINITY, f64::max)
}

#[derive(Debug, Clone, Copy)]
struct ParamSet {
    stencil: StencilUpdateMethod,
    minimization: MinimizationProblemMethod,
    interpolant: InterpolantRepresentation,
}

impl ParamSet {
    #[allow(dead_code)]
    fn label(self) -> String {
        format!(
            "{}/{}/{}",
            self.stencil_label(),
            self.minimization_label(),
            self.interpolant_label()
        )
    }
    fn stencil_label(self) -> String {
        match self.stencil {
            StencilUpdateMethod::Mesh => "Mesh".into(),
            StencilUpdateMethod::Ell1(t) => format!("Ell1({t})"),
            StencilUpdateMethod::MeshEll1(t) => format!("MeshEll1({t})"),
        }
    }
    fn minimization_label(self) -> &'static str {
        match self.minimization {
            MinimizationProblemMethod::FermatIntegral => "FermatIntegral",
            MinimizationProblemMethod::EikonalEquation => "EikonalEquation",
            MinimizationProblemMethod::CellBasedMarching => "CellBasedMarching",
            MinimizationProblemMethod::QuadraticCurve => "QuadraticCurve",
        }
    }
    fn interpolant_label(self) -> &'static str {
        match self.interpolant {
            InterpolantRepresentation::Cubic => "Cubic",
            InterpolantRepresentation::Graph => "Graph",
        }
    }
    fn short(self) -> String {
        let s = match self.stencil {
            StencilUpdateMethod::Mesh => "Mesh".into(),
            StencilUpdateMethod::Ell1(t) => format!("ℓ₁({t:.1})"),
            StencilUpdateMethod::MeshEll1(t) => format!("M+ℓ₁({t:.1})"),
        };
        let m = match self.minimization {
            MinimizationProblemMethod::FermatIntegral => "Fermat",
            MinimizationProblemMethod::EikonalEquation => "Eikonal",
            MinimizationProblemMethod::CellBasedMarching => "Cell",
            MinimizationProblemMethod::QuadraticCurve => "Quad",
        };
        let i = match self.interpolant {
            InterpolantRepresentation::Cubic => "Cubic",
            InterpolantRepresentation::Graph => "Graph",
        };
        format!("{s}/{m}/{i}")
    }
}

const ELL1_THRESHOLDS: &[f64] = &[0.5, 2.0];

fn all_jm_param_sets() -> Vec<ParamSet> {
    let mut stencils = vec![StencilUpdateMethod::Mesh];
    for &t in ELL1_THRESHOLDS {
        stencils.push(StencilUpdateMethod::Ell1(t));
        stencils.push(StencilUpdateMethod::MeshEll1(t));
    }
    let minimizations = [
        MinimizationProblemMethod::FermatIntegral,
        MinimizationProblemMethod::EikonalEquation,
        MinimizationProblemMethod::CellBasedMarching,
        MinimizationProblemMethod::QuadraticCurve,
    ];
    let interpolants = [
        InterpolantRepresentation::Cubic,
        InterpolantRepresentation::Graph,
    ];
    let mut sets = Vec::new();
    for &stencil in &stencils {
        for &minimization in &minimizations {
            for &interpolant in &interpolants {
                sets.push(ParamSet {
                    stencil,
                    minimization,
                    interpolant,
                });
            }
        }
    }
    sets
}

fn fmt_duration(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        return format!("{s}s");
    }
    let m = s / 60;
    let s = s % 60;
    if m < 60 {
        return format!("{m}m{s:02}s");
    }
    let h = m / 60;
    let m = m % 60;
    format!("{h}h{m:02}m{s:02}s")
}

fn main() {
    let jm_params = all_jm_param_sets();
    let n_jm = jm_params.len();

    // Resolve sphere paths — only keep files that actually exist on disk.
    let sphere_paths: Vec<PathBuf> = SPHERE_FILES
        .iter()
        .map(|f| PathBuf::from(SPHERE_DIR).join(f))
        .filter(|p| {
            if p.exists() {
                true
            } else {
                eprintln!(
                    "  {} sphere file not found, skipping: {}",
                    c(YELLOW, "⚠"),
                    p.display()
                );
                false
            }
        })
        .collect();

    let n_spheres = sphere_paths.len();
    if n_spheres == 0 {
        eprintln!(
            "  {} No sphere files found in {SPHERE_DIR}. \
             Run gen_spheres.py first.",
            c(RED, "✘")
        );
        std::process::exit(1);
    }

    let total_runs = n_spheres * (1 + n_jm); // wrong actually but idc

    println!();
    println!(
        "{}",
        c(BOLD, "╔══════════════════════════════════════════════════╗")
    );
    println!(
        "{}",
        c(BOLD, "║     Geopathic Sphere Convergence Benchmark       ║")
    );
    println!(
        "{}",
        c(BOLD, "╚══════════════════════════════════════════════════╝")
    );
    println!();
    println!(
        "  {}  {} sphere meshes",
        c(CYAN, "⬡"),
        c(BOLD, &n_spheres.to_string())
    );
    println!(
        "  {}  {} JM param combos",
        c(CYAN, "⬡"),
        c(BOLD, &n_jm.to_string())
    );
    println!(
        "  {}  {} total runs  {}",
        c(CYAN, "⬡"),
        c(BOLD, &total_runs.to_string()),
        c(
            DIM,
            "(1 FM + n_jm JM per sphere, fixed source = north pole)"
        ),
    );
    println!();
    println!(
        "  {}  Accuracy metric: error vs. spherical arc-length  {}",
        c(YELLOW, "▶"),
        c(DIM, "d(v) = arccos(v_z + 1)"),
    );
    println!();

    let timestamp = chrono::Local::now().format("%m:%d_%H:%M:%S").to_string();
    let csv_path = format!("../benchmarks/csv/sphere_convergence_{timestamp}.csv");
    let mut file = File::create(&csv_path).expect("Cannot create output CSV");

    writeln!(
        file,
        "sphere_file,n_vertices,n_faces,h_mean,\
     method_type,stencil,minimization,interpolant,\
     time_s,mae_vs_gt,max_err_vs_gt,repeats"
    )
    .unwrap();

    {
        let wb = ProgressBar::new(5);
        wb.set_style(
            ProgressStyle::with_template(
                "  {spinner:.cyan} {msg} [{bar:30.yellow/dim}] {pos}/{len}",
            )
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ "),
        );
        wb.set_message(c(DIM, "Warming up…"));
        wb.enable_steady_tick(Duration::from_millis(80));

        let warmup_path = sphere_paths.first().unwrap();
        if let Ok(m) = load_manifold(warmup_path.to_str().unwrap()) {
            for _ in 0..5 {
                let _ = FastMarching::new(&m).compute_distance(Sources::from(0));
                let ps = ParamSet {
                    stencil: StencilUpdateMethod::Mesh,
                    minimization: MinimizationProblemMethod::FermatIntegral,
                    interpolant: InterpolantRepresentation::Cubic,
                };
                let _ = JetMarching::new(&m, UnitSlowness)
                    .with_stencil(ps.stencil)
                    .with_minimization(ps.minimization)
                    .with_interpolant(ps.interpolant)
                    .compute_distance(0usize);
                wb.inc(1);
            }
        }
        wb.finish_with_message(format!("{} Warm-up complete.", c(GREEN, "✔")));
    }
    println!();

    let mp = Arc::new(MultiProgress::new());

    let overall = mp.add(ProgressBar::new(total_runs as u64));
    overall.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {msg}\n  \
             [{bar:50.cyan/blue}] {pos:>6}/{len} runs  \
             {percent:>3}%  elapsed {elapsed_precise}  ETA {eta_precise}",
        )
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ ")
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    overall.enable_steady_tick(Duration::from_millis(100));

    let sphere_bar = mp.add(ProgressBar::new(n_spheres as u64));
    sphere_bar.set_style(
        ProgressStyle::with_template(
            "  {spinner:.magenta} {msg:<48} [{bar:30.magenta/dim}] {pos}/{len} spheres",
        )
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ ")
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    sphere_bar.enable_steady_tick(Duration::from_millis(120));

    let method_bar = mp.add(ProgressBar::new((1 + n_jm) as u64));
    method_bar.set_style(
        ProgressStyle::with_template(
            "    {spinner:.yellow} {msg:<48} [{bar:26.yellow/dim}] {pos}/{len}",
        )
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ ")
        .progress_chars("▪▫ "),
    );
    method_bar.enable_steady_tick(Duration::from_millis(80));

    let bench_start = std::time::Instant::now();

    for (sphere_idx, path) in sphere_paths.iter().enumerate() {
        let file_name = path.file_name().unwrap().to_str().unwrap();
        let short_name = file_name.trim_end_matches(".obj");

        sphere_bar.set_message(format!("{}  {}", c(BOLD, short_name), c(DIM, "loading…"),));
        overall.set_message(format!(
            "{} sphere {}/{}: {}",
            c(CYAN, "▶"),
            sphere_idx + 1,
            n_spheres,
            c(BOLD, short_name),
        ));

        let manifold = match load_manifold(path.to_str().unwrap()) {
            Ok(m) => m,
            Err(e) => {
                sphere_bar.println(format!(
                    "  {} {short_name}: load failed — {e:#?}",
                    c(RED, "✘"),
                ));
                sphere_bar.inc(1);
                overall.inc((1 + n_jm) as u64);
                continue;
            }
        };

        let n_v = manifold.vertices().len();
        let n_f = manifold.faces().len();

        let h_mean = {
            let faces = manifold.faces();
            let verts = &manifold.vertices();
            let mut edges = std::collections::HashSet::new();
            for &face in faces.iter() {
                let (a, b, c_idx) = (face.0, face.1, face.2);
                for (i, j) in [(a, b), (b, c_idx), (a, c_idx)] {
                    edges.insert((i.min(j), i.max(j)));
                }
            }
            let sum: f64 = edges
                .iter()
                .map(|&(i, j): &(usize, usize)| {
                    ((verts[i].clone() as DVector<f64>) - (verts[j].clone() as DVector<f64>)).norm()
                })
                .sum();
            sum / edges.len() as f64
        };

        sphere_bar.set_message(format!(
            "{} {}",
            c(BOLD, short_name),
            c(DIM, &format!("({n_v}v  {n_f}f  h={h_mean:.4})")),
        ));

        let source: usize = manifold
            .vertices()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.norm_squared().partial_cmp(&b.norm_squared()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let gt = ground_truth(&manifold, source);

        method_bar.set_length((1 + n_jm) as u64);
        method_bar.set_position(0);
        method_bar.reset_eta();

        method_bar.set_message(format!("{}", c(CYAN, "FastMarching")));
        {
            let repeats = if n_v < 5_000 { 3 } else { 1 };
            let mut best_elapsed = Duration::MAX;
            let mut last_result = None;
            for _ in 0..repeats {
                let tick = std::time::Instant::now();
                let r = FastMarching::new(&manifold).compute_distance(source);
                let t = tick.elapsed();
                if t < best_elapsed {
                    best_elapsed = t;
                    last_result = Some(r);
                }
            }
            match last_result.unwrap() {
                Err(e) => {
                    overall.println(format!(
                        "  {} FM error on {short_name}: {e}",
                        c(YELLOW, "⚠"),
                    ));
                }
                Ok(d) => {
                    let dist: Vec<f64> = d.iter().copied().collect();
                    let mae_v = mae(&gt, &dist);
                    let merr = max_abs_err(&gt, &dist);
                    writeln!(
                        file,
                        "{file_name},{n_v},{n_f},{h_mean:.8},\
                 FM,,,\
                 {t:.9},{mae:.9},{merr:.9},{repeats}",
                        t = best_elapsed.as_secs_f64(),
                        mae = mae_v,
                        merr = merr,
                    )
                    .unwrap();
                }
            }
        }
        method_bar.inc(1);
        overall.inc(1);

        for &params in jm_params.iter() {
            method_bar.set_message(format!("{}", c(MAGENTA, &params.short())));

            let tick = std::time::Instant::now();
            let result = JetMarching::new(&manifold, UnitSlowness)
                .with_stencil(params.stencil)
                .with_minimization(params.minimization)
                .with_interpolant(params.interpolant)
                .compute_distance(source);
            let elapsed = tick.elapsed();

            match result {
                Err(_) => {
                    overall.println(format!(
                        "  {} JM failed: {short_name} {}",
                        c(YELLOW, "⚠"),
                        params.short(),
                    ));
                }
                Ok((d, _amplitudes)) => {
                    let dist: Vec<f64> = d.iter().copied().collect();
                    let mae_v = mae(&gt, &dist);
                    let merr = max_abs_err(&gt, &dist);
                    writeln!(
                        file,
                        "{file_name},{n_v},{n_f},{h_mean:.8},\
                         JM,{stencil},{min},{interp},\
                         {t:.9},{mae:.9},{merr:.9},1",
                        stencil = params.stencil_label(),
                        min = params.minimization_label(),
                        interp = params.interpolant_label(),
                        t = elapsed.as_secs_f64(),
                        mae = mae_v,
                        merr = merr,
                    )
                    .unwrap();
                }
            }

            method_bar.inc(1);
            overall.inc(1);
        }

        let elapsed_total = bench_start.elapsed();
        let done = overall.position() as f64;
        let rate = done / elapsed_total.as_secs_f64();
        let remaining = total_runs as f64 - done;
        let eta = if rate > 0.0 {
            Duration::from_secs_f64(remaining / rate)
        } else {
            Duration::MAX
        };
        overall.set_message(format!(
            "{} {short_name}  {}  ETA {}",
            c(CYAN, "▶"),
            c(DIM, &format!("{rate:.1} runs/s")),
            c(YELLOW, &fmt_duration(eta)),
        ));

        sphere_bar.inc(1);
        overall.println(format!(
            "  {} {}{} — {n_v} verts  {n_f} faces  h={h_mean:.4}",
            c(GREEN, "✔"),
            c(BOLD, short_name),
            c(DIM, ".obj"),
        ));
    }

    method_bar.finish_and_clear();
    sphere_bar.finish_with_message(format!("{} All spheres complete.", c(GREEN, "✔")));
    overall.finish_with_message(format!(
        "{} Benchmark finished in {}.",
        c(GREEN, "✔"),
        c(BOLD, &fmt_duration(bench_start.elapsed())),
    ));

    println!();
    println!(
        "{}",
        c(
            GREEN,
            "╔═══════════════════════════════════════════════════════╗"
        )
    );
    println!("{}", c(GREEN, &format!("║  Results → {csv_path:<43}║")));
    println!(
        "{}",
        c(
            GREEN,
            "╚═══════════════════════════════════════════════════════╝"
        )
    );
}
