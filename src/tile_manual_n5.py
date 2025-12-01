"""
Tile a manual n=5 configuration to generate full submission.

This takes your hand-optimized n=5 config and tiles it for all n values.

Usage:
    python tile_manual_n5.py --n5-file manual_n5_box1.50.txt --output ../submissions/manual_tiled.csv
"""
import sys
from pathlib import Path
from loguru import logger
import argparse
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent))

from tree_geometry import ChristmasTree
from utils import calculate_bounding_square, calculate_total_score, export_submission


def load_base_config(filepath):
    """Load base configuration from file (works for any n)."""
    logger.info(f"Loading base configuration from: {filepath}")
    
    positions = []
    angles = []
    base_n = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            # Parse: 002_0,s0.123,s0.456,s45.0
            parts = line.split(',')
            if len(parts) == 4:
                # Extract n from id (e.g., "002_0" -> n=2)
                tree_id = parts[0]
                n_str = tree_id.split('_')[0]
                current_n = int(n_str)
                
                if base_n is None:
                    base_n = current_n
                
                x = float(parts[1][1:])  # Remove 's' prefix
                y = float(parts[2][1:])
                deg = float(parts[3][1:])
                
                positions.append([x, y])
                angles.append(deg)
    
    if len(positions) != base_n:
        logger.error(f"Expected {base_n} trees, got {len(positions)}")
        return None, None, None
    
    # Create trees
    trees = []
    for i in range(base_n):
        tree = ChristmasTree(str(positions[i][0]), str(positions[i][1]), str(angles[i]))
        trees.append(tree)
    
    # Calculate score
    side = calculate_bounding_square(trees)
    score = float(side ** 2) / base_n
    
    logger.info(f"✓ Loaded n={base_n} configuration")
    logger.info(f"  Side length: {float(side):.6f}")
    logger.info(f"  Score: {score:.6f}")
    logger.info("")
    
    return base_n, positions, angles


def tile_base_configuration(base_n, base_positions, base_angles, n, layout='grid'):
    """
    Create n-tree configuration by tiling the base cluster.
    
    Args:
        base_n: Size of base cluster (e.g., 2, 3, 5)
        base_positions: List of [x, y] for base_n trees
        base_angles: List of angles for base_n trees
        n: Target number of trees
        layout: 'grid' or 'linear' tiling pattern
    
    Returns:
        List of n trees
    """
    if n <= base_n:
        # For n <= base_n, just use subset
        trees = []
        for i in range(n):
            tree = ChristmasTree(
                str(base_positions[i][0]),
                str(base_positions[i][1]),
                str(base_angles[i])
            )
            trees.append(tree)
        return trees
    
    # Calculate base cluster bounding box
    base_trees = []
    for i in range(base_n):
        tree = ChristmasTree(str(base_positions[i][0]), str(base_positions[i][1]), str(base_angles[i]))
        base_trees.append(tree)
    
    base_side = float(calculate_bounding_square(base_trees))
    
    # Calculate grid dimensions
    num_clusters = int(np.ceil(n / base_n))
    
    if layout == 'linear':
        # Place all clusters in a single row
        k_x = num_clusters
        k_y = 1
    else:
        # Optimal rectangular grid (not square)
        # Try to find rectangle closest to square that fits num_clusters
        best_ratio = float('inf')
        best_k_x, best_k_y = 1, num_clusters
        
        for k_x in range(1, num_clusters + 1):
            k_y = int(np.ceil(num_clusters / k_x))
            ratio = max(k_x, k_y) / min(k_x, k_y)  # How far from square
            if ratio < best_ratio:
                best_ratio = ratio
                best_k_x, best_k_y = k_x, k_y
        
        k_x, k_y = best_k_x, best_k_y
    
    # Tile spacing - use exact bounding box size (no gap!)
    spacing = base_side * 1.00
    
    # Generate tiled trees
    all_trees = []
    cluster_count = 0
    
    for i in range(k_x):
        for j in range(k_y):
            if len(all_trees) >= n:
                break
            
            # Offset for this cluster
            offset_x = i * spacing
            offset_y = j * spacing
            
            # Add trees from this cluster
            for tree_idx in range(base_n):
                if len(all_trees) >= n:
                    break
                
                new_x = base_positions[tree_idx][0] + offset_x
                new_y = base_positions[tree_idx][1] + offset_y
                
                tree = ChristmasTree(
                    str(new_x),
                    str(new_y),
                    str(base_angles[tree_idx])
                )
                all_trees.append(tree)
            
            cluster_count += 1
        
        if len(all_trees) >= n:
            break
    
    return all_trees[:n]


def generate_tiled_submission(base_file, output_csv, layout='grid', verbose=True):
    """
    Generate full submission by tiling base configuration.
    
    Args:
        base_file: Path to manual base configuration file (any n)
        output_csv: Output submission path
        layout: 'grid' or 'linear' tiling pattern
        verbose: Print progress
    
    Returns:
        Total score
    """
    # Load base config
    base_n, base_positions, base_angles = load_base_config(base_file)
    
    if base_positions is None:
        return None
    
    if verbose:
        logger.info("="*80)
        logger.info(f"GENERATING TILED SUBMISSION (base n={base_n}, layout={layout})")
        logger.info("="*80)
        logger.info("")
    
    start_time = time.time()
    
    all_tree_data = []
    side_lengths = []
    
    for n in range(1, 201):
        # Generate configuration by tiling
        trees = tile_base_configuration(base_n, base_positions, base_angles, n, layout=layout)
        
        # Calculate side length
        side = calculate_bounding_square(trees)
        side_lengths.append(side)
        
        # Store tree data
        for tree in trees:
            all_tree_data.append([tree.center_x, tree.center_y, tree.angle])
        
        # Progress update
        if verbose and n % 20 == 0:
            elapsed = time.time() - start_time
            current_score = calculate_total_score(side_lengths)
            score_n = float(side ** 2) / n
            
            logger.info(f"n={n:3d}: side={float(side):.4f}, score={score_n:.6f}")
            logger.info(f"  Cumulative score: {current_score:.6f}")
            logger.info(f"  Time: {elapsed:.1f}s")
            logger.info("")
    
    # Export submission
    export_submission(all_tree_data, output_csv)
    
    # Calculate final score
    total_score = calculate_total_score(side_lengths)
    
    elapsed = time.time() - start_time
    
    if verbose:
        logger.info("="*80)
        logger.info("FINAL RESULTS")
        logger.info("="*80)
        logger.info(f"Base cluster: n={base_n}")
        logger.info(f"Total score: {total_score:.6f}")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info("")
        logger.info(f"✓ Saved to: {output_csv}")
        logger.info("")
    
    return total_score


def main():
    parser = argparse.ArgumentParser(description='Tile manual configuration (works with any n)')
    parser.add_argument('--base-file', type=str, required=True,
                       help='Path to manual base configuration file (e.g., manual_n2_box0.99.txt)')
    parser.add_argument('--output', type=str, default='../submissions/manual_tiled.csv',
                       help='Output submission CSV')
    parser.add_argument('--layout', type=str, default='grid', choices=['grid', 'linear'],
                       help='Tiling layout: grid (optimal rectangle) or linear (single row)')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.base_file).exists():
        logger.error(f"File not found: {args.base_file}")
        return
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate submission
    total_score = generate_tiled_submission(args.base_file, args.output, layout=args.layout, verbose=True)
    
    if total_score:
        # Save score to log
        scores_file = Path(args.output).parent / 'scores.txt'
        with open(scores_file, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{Path(args.output).name}: {float(total_score):.6f} (manual_tiled) - {timestamp}\n")
        
        logger.info(f"✓ Score logged to: {scores_file}")
        logger.info("")
        
        # Summary
        logger.info("Next steps:")
        logger.info("  1. Submit to Kaggle")
        logger.info("  2. If score is good, optimize base cluster further")
        logger.info("  3. Try different base sizes (n=2, n=3, n=4, n=5)")
        logger.info("  4. Consider backward iteration on this submission")


if __name__ == "__main__":
    main()