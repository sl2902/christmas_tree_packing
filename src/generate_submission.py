"""
Main script to generate Kaggle submission.

Usage:
    python generate_submission.py [options]

Options:
    --method: 'rotation' (default) or 'full_rotation'
    --output: Output filename (default: submission.csv)
    --seed: Random seed for reproducibility
"""
import sys
import random
import time
import argparse
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from placement import greedy_placement
from optimization import rotation_optimization, full_rotation_optimization
from simulated_annealing import hybrid_optimization, simulated_annealing_optimization
from utils import calculate_total_score, export_submission


def generate_solution(method='rotation', seed=42, verbose=True):
    """
    Generate complete solution for all 200 configurations.
    
    Args:
        method: Optimization method ('rotation' or 'full_rotation')
        seed: Random seed
        verbose: print progress
    
    Returns:
        Tuple of (all_tree_data, side_lengths, total_score)
    """
    random.seed(seed)
    
    if verbose:
        logger.info("="*80)
        logger.info(f"GENERATING SOLUTION - Method: {method}")
        logger.info("="*80)
        logger.info('')
    
    all_tree_data = []
    side_lengths = []
    current_placed_trees = []
    start_time = time.time()
    
    for n in range(1, 201):
        # Greedy placement
        current_placed_trees, baseline_side = greedy_placement(
            n, existing_trees=current_placed_trees
        )
        
        # Optimization
        if method == 'rotation':
            current_placed_trees, optimized_side = rotation_optimization(
                current_placed_trees
            )
        elif method == 'full_rotation':
            current_placed_trees, optimized_side = full_rotation_optimization(
                current_placed_trees
            )
        elif method == 'sa':
            current_placed_trees, optimized_side = simulated_annealing_optimization(
                current_placed_trees,
                verbose=(n % 50 == 0)  # Show progress every 50 configs
            )
        elif method == 'hybrid':
            current_placed_trees, optimized_side = hybrid_optimization(
                current_placed_trees,
                verbose=(n % 50 == 0)
            )
        else:
            # No optimization
            optimized_side = baseline_side
        
        side_lengths.append(optimized_side)
        
        # Store all trees for this configuration
        for tree in current_placed_trees:
            all_tree_data.append([tree.center_x, tree.center_y, tree.angle])
        
        # Progress update
        if verbose and n % 20 == 0:
            elapsed = time.time() - start_time
            current_score = calculate_total_score(side_lengths)
            
            logger.info(f"Progress: {n}/200 trees ({n/2:.0f}% complete)")
            logger.info(f"  Time elapsed: {elapsed:.1f}s")
            logger.info(f"  Current score: {current_score:.6f}")
            logger.info(f"  Avg time per config: {elapsed/n:.2f}s")
            logger.info('')
    
    # Calculate final score
    total_score = calculate_total_score(side_lengths)
    
    if verbose:
        logger.info("="*80)
        logger.info("FINAL RESULTS")
        logger.info("="*80)
        logger.info(f"Final score: {total_score:.6f}")
        logger.info(f"Total time: {time.time() - start_time:.1f}s")
        logger.info('')
    
    return all_tree_data, side_lengths, total_score


def main():
    parser = argparse.ArgumentParser(description='Generate Christmas Tree Packing submission')
    parser.add_argument('--method', type=str, default='rotation',
                       choices=['rotation', 'full_rotation', 'baseline', 'sa', 'hybrid'],
                       help='Optimization method')
    parser.add_argument('--output', type=str, default='../submissions/submission.csv',
                       help='Output filename')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Generate solution
    all_tree_data, side_lengths, total_score = generate_solution(
        method=args.method,
        seed=args.seed,
        verbose=True
    )
    
    # Export submission
    export_submission(all_tree_data, args.output)
    
    # Save score
    scores_file = Path(args.output).parent / 'scores.txt'
    with open(scores_file, 'a') as f:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{Path(args.output).name}: {float(total_score):.6f} ({args.method}) - {timestamp}\n")
    
    logger.info(f"Score appended to: {scores_file}")


if __name__ == "__main__":
    main()