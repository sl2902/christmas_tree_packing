// src/main.rs
// Rust SA using the only proven accurate method: O(N^2) brute-force geometric check.
// This guarantees best_score matches the final validation.

use geo::{Coord, LineString, Polygon, BoundingRect};
use geo::algorithm::intersects::Intersects;
use geo::algorithm::rotate::Rotate;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::io::{BufWriter, Write};
use clap::Parser;
use std::time::Instant;

// --- GEOMETRY DEFINITIONS ---

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Test single n value
    #[arg(short, long)]
    n: Option<usize>,
    
    /// Start from this n
    #[arg(long)]
    start_n: Option<usize>,
    
    /// Number of iterations
    #[arg(long, default_value_t = 50000)]
    iterations: usize,
    
    /// Number of trials per n
    #[arg(long, default_value_t = 5)]
    trials: usize,
    
    /// Output CSV file
    #[arg(long)]
    output: Option<String>,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
    
    /// Full submission (n=1-200)
    #[arg(long)]
    full_submission: bool,
}

// Christmas tree geometry
fn create_tree_polygon(center_x: f64, center_y: f64, angle_deg: f64) -> Polygon<f64> {
    let trunk_w = 0.15;
    let trunk_h = 0.2;
    let base_w = 0.7;
    let mid_w = 0.4;
    let top_w = 0.25;
    let tip_y = 0.8;
    let tier1 = 0.5;
    let tier2 = 0.25;
    let trunk_b = -trunk_h;
    
    let points = vec![
        (0.0, tip_y),
        (top_w/2.0, tier1), (top_w/4.0, tier1),
        (mid_w/2.0, tier2), (mid_w/4.0, tier2),
        (base_w/2.0, 0.0),
        (trunk_w/2.0, 0.0), (trunk_w/2.0, trunk_b),
        (-trunk_w/2.0, trunk_b), (-trunk_w/2.0, 0.0),
        (-base_w/2.0, 0.0),
        (-mid_w/4.0, tier2), (-mid_w/2.0, tier2),
        (-top_w/4.0, tier1), (-top_w/2.0, tier1),
    ];
    
    let coords: Vec<Coord<f64>> = points.iter().map(|&(x, y)| Coord { x, y }).collect();
    let mut poly = Polygon::new(LineString::from(coords), vec![]);
    poly = poly.rotate_around_point(angle_deg, geo::Point::new(0.0, 0.0));
    
    let translated: Vec<Coord<f64>> = poly.exterior().coords()
        .map(|c| Coord { x: c.x + center_x, y: c.y + center_y })
        .collect();
    
    Polygon::new(LineString::from(translated), vec![])
}

#[derive(Clone)]
struct Tree {
    x: f64,
    y: f64,
    angle: f64,
    polygon: Polygon<f64>,
}

impl Tree {
    fn new(x: f64, y: f64, angle: f64) -> Self {
        let polygon = create_tree_polygon(x, y, angle);
        Tree { x, y, angle, polygon }
    }
}

fn bounding_square(trees: &[Tree]) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    
    for tree in trees {
        if let Some(rect) = tree.polygon.bounding_rect() {
            min_x = min_x.min(rect.min().x);
            max_x = max_x.max(rect.max().x);
            min_y = min_y.min(rect.min().y);
            max_y = max_y.max(rect.max().y);
        }
    }
    
    (max_x - min_x).max(max_y - min_y)
}

// The only accurate score function (O(N^2) brute-force)
fn calculate_accurate_score(trees: &[Tree], n: usize) -> f64 {
    use geo::algorithm::area::Area;
    use geo::algorithm::bool_ops::BooleanOps;
    use geo::algorithm::relate::Relate;
    
    let mut penalty = 0.0;
    
    // Check ALL pairs for collision (O(N^2))
    for i in 0..trees.len() {
        for j in (i+1)..trees.len() {
            let intersects = trees[i].polygon.intersects(&trees[j].polygon);
            let touches = trees[i].polygon.relate(&trees[j].polygon).is_touches();
            
            if intersects && !touches {
                let intersection = trees[i].polygon.intersection(&trees[j].polygon);
                let overlap_area = intersection.unsigned_area();
                penalty += overlap_area * 1000.0;
            }
        }
    }
    
    // Calculate bounding square with ALL trees
    let side = bounding_square(trees);
    (side * side) / (n as f64) + penalty
}

fn count_overlaps(trees: &[Tree]) -> usize {
    use geo::algorithm::relate::Relate;
    let mut count = 0;
    for i in 0..trees.len() {
        for j in (i+1)..trees.len() {
            let intersects = trees[i].polygon.intersects(&trees[j].polygon);
            let touches = trees[i].polygon.relate(&trees[j].polygon).is_touches();
            if intersects && !touches {
                count += 1;
            }
        }
    }
    count
}

// --- SIMULATED ANNEALING ---

fn simulated_annealing(n: usize, iterations: usize, seed: u64, verbose: bool) -> (Vec<Tree>, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Initialization
    let radius = 2.0;
    let mut trees: Vec<Tree> = (0..n)
        .map(|_| {
            let r = radius * rng.gen::<f64>().sqrt();
            let theta = 2.0 * std::f64::consts::PI * rng.gen::<f64>();
            let x = r * theta.cos();
            let y = r * theta.sin();
            let angle = rng.gen::<f64>() * 360.0;
            Tree::new(x, y, angle)
        })
        .collect();
    
    let mut best_trees = trees.clone();
    let mut best_positions: Vec<(f64, f64, f64)> = trees.iter()
        .map(|t| (t.x, t.y, t.angle))
        .collect();
    
    // Use the only reliable score calculation
    let initial_score = calculate_accurate_score(&trees, n);
    
    let mut best_score = initial_score;
    let mut current_score = initial_score;
    
    // Temperature schedule (matching Python: T0=10.0, T1=0.01)
    let t0: f64 = 10.0;
    let t1: f64 = 0.01;
    
    if verbose {
        eprintln!("  Initial score: {:.6}", current_score);
        eprintln!("  Bounding square: {:.6}", bounding_square(&trees));
    }
    
    let mut accepts = 0;
    
    for iter in 0..iterations {
        let progress = iter as f64 / iterations as f64;
        // Strict exponential cooling schedule (no tuning factor)
        let temperature = t0 * (t1 / t0).powf(progress);
        
        let tree_idx = rng.gen_range(0..n);
        let old_tree = trees[tree_idx].clone();
        
        // Random move (Gaussian/Uniform logic)
        let move_type = rng.gen_range(0..3);
        let (new_x, new_y, new_angle) = match move_type {
            0 => {
                // Translate (GAUSSIAN)
                let step_size = 0.5 * (1.0 - progress * 0.8);
                let normal = Normal::new(0.0, step_size).unwrap();
                let dx = normal.sample(&mut rng);
                let dy = normal.sample(&mut rng);
                (old_tree.x + dx, old_tree.y + dy, old_tree.angle)
            },
            1 => {
                // Rotate (UNIFORM)
                let angle_step = 30.0 * (1.0 - progress * 0.8);
                let da = rng.gen::<f64>() * 2.0 * angle_step - angle_step; 
                (old_tree.x, old_tree.y, (old_tree.angle + da) % 360.0)
            },
            _ => {
                // Both (GAUSSIAN for position, UNIFORM for angle)
                let step_size = 0.3 * (1.0 - progress * 0.8);
                let normal = Normal::new(0.0, step_size).unwrap();
                let dx = normal.sample(&mut rng);
                let dy = normal.sample(&mut rng);
                
                let angle_step = 20.0 * (1.0 - progress * 0.8);
                let da = rng.gen::<f64>() * 2.0 * angle_step - angle_step;
                (old_tree.x + dx, old_tree.y + dy, (old_tree.angle + da) % 360.0)
            }
        };
        
        // Update tree temporarily
        trees[tree_idx] = Tree::new(new_x, new_y, new_angle);
        
        // Calculate new score using the ACCURATE O(N^2) check
        let new_score = calculate_accurate_score(&trees, n);
        
        let delta = new_score - current_score;
        
        // Accept or reject
        if delta < 0.0 || rng.gen::<f64>() < (-delta / temperature).exp() {
            accepts += 1;
            current_score = new_score;
            if new_score < best_score {
                best_score = new_score;
                // Save current positions/angles
                for i in 0..trees.len() {
                    best_positions[i] = (trees[i].x, trees[i].y, trees[i].angle);
                }
            }
        } else {
            // Restore old tree
            trees[tree_idx] = old_tree;
        }
        
        if verbose && iter > 0 && iter % 10000 == 0 {
            let accept_rate = accepts as f64 / iter as f64;
            let overlaps = count_overlaps(&trees);
            eprintln!("  Iter {:6}: T={:.4}, current={:.6}, best={:.6}, accept={:.3}, overlaps={}", 
                      iter, temperature, current_score, best_score, accept_rate, overlaps);
        }
    }
    
    // Reconstruct best_trees from saved positions
    for i in 0..best_positions.len() {
        let (x, y, angle) = best_positions[i];
        best_trees[i] = Tree::new(x, y, angle);
    }
    
    if verbose {
        eprintln!("  Final best score: {:.6}", best_score);
        
        // Final validation using the ACCURATE O(N^2) score check
        let validation_score = calculate_accurate_score(&best_trees, n);
        
        // Recalculate overlap count
        let overlap_count = count_overlaps(&best_trees);
        let side = bounding_square(&best_trees);
        
        eprintln!("  VALIDATION: {} overlaps, side={:.6}", overlap_count, side); 
        
        eprintln!("  VALIDATION: Recalculated score = {:.6}", validation_score);
        if (validation_score - best_score).abs() > 0.001 {
            eprintln!("  WARNING: Score mismatch! best_score={:.6} but validation={:.6}", 
                      best_score, validation_score);
        }
    }
    
    (best_trees, best_score)
}

// --- MAIN EXECUTION ---

fn optimize_with_trials(n: usize, iterations: usize, trials: usize, verbose: bool) -> (Vec<Tree>, f64) {
    if verbose {
        eprintln!("{}", "=".repeat(80));
        eprintln!("OPTIMIZED SIMULATED ANNEALING: n={}, trials={}", n, trials);
        eprintln!("{}", "=".repeat(80));
        eprintln!();
    }
    
    let mut best_trees = Vec::new();
    let mut best_score = f64::INFINITY;
    
    for trial in 0..trials {
        if verbose {
            eprintln!("Trial {}/{}", trial + 1, trials);
        }
        
        // Using a different seed for each trial
        let (trees, score) = simulated_annealing(n, iterations, 42 + trial as u64, verbose);
        
        if score < best_score {
            best_score = score;
            best_trees = trees;
        }
        
        if verbose {
            eprintln!();
        }
    }
    
    if verbose {
        eprintln!("{}", "=".repeat(80));
        eprintln!("BEST SCORE ACROSS {} TRIALS: {:.6}", trials, best_score);
        eprintln!("{}", "=".repeat(80));
        eprintln!();
    }
    
    (best_trees, best_score)
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    
    // Determine range
    let (start_n, end_n) = if let Some(n) = args.n {
        (n, n)
    } else if args.full_submission {
        (1, 200)
    } else if let Some(start) = args.start_n {
        (start, 200)
    } else {
        eprintln!("Error: Must specify --n, --full-submission, or --start-n");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  Test n=5:           cargo run --release -- --n 5");
        eprintln!("  Full submission:    cargo run --release -- --full-submission");
        eprintln!("  Continue from n=77: cargo run --release -- --start-n 77");
        std::process::exit(1);
    };
    
    // Generate dynamic output filename if not specified
    let output = if let Some(out) = args.output {
        out
    } else {
        if args.n.is_some() {
            format!("./submissions/rust_sa_i{}_t{}_n{}.csv", 
                    args.iterations, args.trials, args.n.unwrap())
        } else if args.full_submission {
            format!("./submissions/rust_sa_i{}_t{}_n1-200.csv", 
                    args.iterations, args.trials)
        } else {
            format!("./submissions/rust_sa_i{}_t{}_n{}-200.csv", 
                    args.iterations, args.trials, start_n)
        }
    };
    
    // Create output directory if needed
    if let Some(parent) = std::path::Path::new(&output).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    
    if !args.verbose {
        eprintln!("{}", "=".repeat(80));
        eprintln!("RUST SIMULATED ANNEALING (O(N^2) ACCURATE)");
        eprintln!("{}", "=".repeat(80));
        eprintln!();
    }
    
    let start_time = Instant::now();
    let mut all_tree_data: Vec<(f64, f64, f64)> = Vec::new();
    
    for n in start_n..=end_n {
        let iters = args.iterations;
        
        if !args.verbose {
            eprintln!("Optimizing n={} ({} iterations, {} trials)...", n, iters, args.trials);
        }
        
        let (trees, score) = optimize_with_trials(n, iters, args.trials, args.verbose);
        
        for tree in &trees {
            all_tree_data.push((tree.x, tree.y, tree.angle));
        }
        
        if !args.verbose {
            eprintln!("  Final score: {:.6}", score);
            
            if n % 5 == 0 {
                let elapsed = start_time.elapsed().as_secs();
                let n_processed = n - start_n + 1;
                let time_per_n = elapsed as f64 / n_processed as f64;
                let estimated_remaining_secs = (end_n - n) as f64 * time_per_n;
                eprintln!("  Time elapsed: {} min", elapsed / 60);
                eprintln!("  Estimated remaining: {:.1} min", estimated_remaining_secs / 60.0);
            }
            eprintln!();
        }
    }
    
    // Export CSV
    let file = File::create(&output)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "id,x,y,rotation")?;
    
    let mut idx = 0;
    for n in start_n..=end_n {
        for i in 0..n {
            let (x, y, angle) = all_tree_data[idx];
            writeln!(writer, "{:03}_{},s{:.6},s{:.6},s{:.1}", n, i, x, y, angle)?;
            idx += 1;
        }
    }
    
    let elapsed = start_time.elapsed();
    eprintln!();
    eprintln!("{}", "=".repeat(80));
    eprintln!("FINAL RESULTS");
    eprintln!("{}", "=".repeat(80));
    eprintln!("Total time: {:.1} minutes", elapsed.as_secs_f64() / 60.0);
    eprintln!("✓ Saved to: {}", output);
    
    // If single n test, show tree positions
    if let Some(n) = args.n {
        eprintln!();
        eprintln!("Tree positions:");
        for i in 0..n {
            let (x, y, angle) = all_tree_data[i];
            eprintln!("  {:03}_{},s{:.6},s{:.6},s{:.1}", n, i, x, y, angle);
        }
    }
    
    Ok(())
}