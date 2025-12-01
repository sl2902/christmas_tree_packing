"""
Optimized Simulated Annealing for tree packing.

Implements all best practices:
- Fast iteration loop
- Incremental collision checking (O(N) not O(N²))
- Exponential cooling schedule
- Multiple neighborhood moves
- Best solution tracking
- Multiple random seeds

Usage:
    python optimized_sa.py --n 5 --iterations 100000 --trials 10
    python optimized_sa.py --full-submission --output ../submissions/sa_submission.csv
"""
import sys
from pathlib import Path
import numpy as np
import random
import time
from loguru import logger
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from tree_geometry import ChristmasTree
from utils import calculate_bounding_square, calculate_total_score, export_submission
from shapely.strtree import STRtree


class FastTreePacker:
    """Fast simulated annealing for tree packing."""
    
    def __init__(self, n, seed=42):
        self.n = n
        random.seed(seed)
        np.random.seed(seed)
        
        # State: positions and angles
        self.positions = np.zeros((n, 2), dtype=np.float64)
        self.angles = np.zeros(n, dtype=np.float64)
        
        # Initialize randomly in small area
        radius = 2.0
        for i in range(n):
            r = radius * np.sqrt(random.random())
            theta = 2 * np.pi * random.random()
            self.positions[i, 0] = r * np.cos(theta)
            self.positions[i, 1] = r * np.sin(theta)
            self.angles[i] = random.random() * 360
        
        # Trees cache
        self.trees = self._create_trees()
        self.current_score = self._calculate_score()
        
        # Best solution tracking
        self.best_positions = self.positions.copy()
        self.best_angles = self.angles.copy()
        self.best_score = self.current_score
    
    def _create_trees(self):
        """Create tree objects from current state."""
        trees = []
        for i in range(self.n):
            tree = ChristmasTree(
                str(self.positions[i, 0]),
                str(self.positions[i, 1]),
                str(self.angles[i])
            )
            trees.append(tree)
        return trees
    
    def _calculate_score(self):
        """Calculate score for current configuration."""
        trees = self._create_trees()
        
        # Check overlaps (penalty)
        penalty = 0.0
        polygons = [t.polygon for t in trees]
        tree_index = STRtree(polygons)
        
        for i, poly in enumerate(polygons):
            indices = tree_index.query(poly)
            for idx in indices:
                if idx <= i:
                    continue
                if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                    # Heavy penalty for overlaps
                    overlap_area = poly.intersection(polygons[idx]).area / 1e30
                    penalty += overlap_area * 1000
        
        # Bounding square
        side = calculate_bounding_square(trees)
        base_score = float(side ** 2) / self.n
        
        return base_score + penalty
    
    def _calculate_score_after_move(self, tree_idx, new_pos, new_angle):
        """
        Calculate score after moving one tree.
        OPTIMIZED: Only check collisions for moved tree (O(N) not O(N²))
        """
        # Temporarily update
        old_pos = self.positions[tree_idx].copy()
        old_angle = self.angles[tree_idx]
        
        self.positions[tree_idx] = new_pos
        self.angles[tree_idx] = new_angle
        
        # Create trees
        trees = self._create_trees()
        
        # Check overlaps - only for moved tree
        penalty = 0.0
        moved_poly = trees[tree_idx].polygon
        
        for i in range(self.n):
            if i == tree_idx:
                continue
            
            other_poly = trees[i].polygon
            if moved_poly.intersects(other_poly) and not moved_poly.touches(other_poly):
                overlap_area = moved_poly.intersection(other_poly).area / 1e30
                penalty += overlap_area * 1000
        
        # Bounding square
        side = calculate_bounding_square(trees)
        score = float(side ** 2) / self.n + penalty
        
        # Restore
        self.positions[tree_idx] = old_pos
        self.angles[tree_idx] = old_angle
        
        return score
    
    def _get_temperature(self, progress, T0=10.0, T1=0.01):
        """
        Exponential cooling schedule.
        progress: 0.0 to 1.0
        Returns: temperature
        """
        return T0 * (T1 / T0) ** progress
    
    def _accept_move(self, delta, temperature):
        """Acceptance probability for SA."""
        if delta < 0:
            return True  # Always accept improvements
        return random.random() < np.exp(-delta / temperature)
    
    def optimize(self, max_iterations=100000, T0=10.0, T1=0.01, verbose=True):
        """
        Run simulated annealing optimization.
        
        Args:
            max_iterations: Number of iterations
            T0: Initial temperature
            T1: Final temperature
            verbose: Print progress
        """
        if verbose:
            logger.info(f"SA optimization: n={self.n}, iterations={max_iterations}")
            logger.info(f"  Initial score: {self.current_score:.6f}")
        
        for iteration in range(max_iterations):
            progress = iteration / max_iterations
            temperature = self._get_temperature(progress, T0, T1)
            
            # Choose move type randomly
            move_type = random.choice(['translate', 'rotate', 'both'])
            
            # Select random tree
            tree_idx = random.randint(0, self.n - 1)
            
            # Generate neighbor
            if move_type == 'translate':
                # Move position
                step_size = 0.5 * (1 - progress * 0.8)  # Decrease step size over time
                delta = np.random.randn(2) * step_size
                new_pos = self.positions[tree_idx] + delta
                new_angle = self.angles[tree_idx]
                
            elif move_type == 'rotate':
                # Rotate
                angle_step = 30 * (1 - progress * 0.8)
                delta_angle = random.uniform(-angle_step, angle_step)
                new_pos = self.positions[tree_idx].copy()
                new_angle = (self.angles[tree_idx] + delta_angle) % 360
                
            else:  # both
                # Both translate and rotate
                step_size = 0.3 * (1 - progress * 0.8)
                delta = np.random.randn(2) * step_size
                new_pos = self.positions[tree_idx] + delta
                angle_step = 20 * (1 - progress * 0.8)
                delta_angle = random.uniform(-angle_step, angle_step)
                new_angle = (self.angles[tree_idx] + delta_angle) % 360
            
            # Calculate new score (O(N) collision check)
            new_score = self._calculate_score_after_move(tree_idx, new_pos, new_angle)
            
            # Accept or reject
            delta_score = new_score - self.current_score
            
            if self._accept_move(delta_score, temperature):
                # Accept move
                self.positions[tree_idx] = new_pos
                self.angles[tree_idx] = new_angle
                self.current_score = new_score
                
                # Update best
                if new_score < self.best_score:
                    self.best_score = new_score
                    self.best_positions = self.positions.copy()
                    self.best_angles = self.angles.copy()
            
            # Progress logging
            if verbose and iteration % 10000 == 0:
                logger.info(f"  Iter {iteration:6d}: T={temperature:.4f}, "
                           f"current={self.current_score:.6f}, best={self.best_score:.6f}")
        
        # Restore best solution
        self.positions = self.best_positions.copy()
        self.angles = self.best_angles.copy()
        self.current_score = self.best_score
        
        if verbose:
            logger.info(f"  Final best score: {self.best_score:.6f}")
        
        return self.best_score
    
    def get_trees(self):
        """Get final tree configuration."""
        return self._create_trees()


def optimize_with_multiple_trials(n, iterations=100000, trials=10, verbose=True):
    """
    Run SA multiple times with different seeds and keep best.
    
    Args:
        n: Number of trees
        iterations: Iterations per trial
        trials: Number of random trials
        verbose: Print progress
    
    Returns:
        Tuple of (best_trees, best_score)
    """
    if verbose:
        logger.info("="*80)
        logger.info(f"OPTIMIZED SIMULATED ANNEALING: n={n}, trials={trials}")
        logger.info("="*80)
        logger.info("")
    
    best_overall_trees = None
    best_overall_score = float('inf')
    
    for trial in range(trials):
        if verbose:
            logger.info(f"Trial {trial + 1}/{trials}")
        
        packer = FastTreePacker(n, seed=42 + trial)
        score = packer.optimize(max_iterations=iterations, verbose=verbose)
        trees = packer.get_trees()
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_trees = trees
        
        if verbose:
            logger.info("")
    
    if verbose:
        logger.info("="*80)
        logger.info(f"BEST SCORE ACROSS {trials} TRIALS: {best_overall_score:.6f}")
        logger.info("="*80)
        logger.info("")
    
    return best_overall_trees, best_overall_score


def generate_sa_submission(output_csv, 
                           iterations_small=100000,
                           iterations_large=50000,
                           trials=5,
                           checkpoint_file=None,
                           verbose=True):
    """
    Generate full submission using optimized SA with checkpointing.
    
    Args:
        output_csv: Output path
        iterations_small: Iterations for n <= 20
        iterations_large: Iterations for n > 20
        trials: Trials per n
        checkpoint_file: Path to checkpoint file (auto-created if None)
        verbose: Print progress
    """
    # Setup checkpoint
    if checkpoint_file is None:
        checkpoint_file = output_csv.replace('.csv', '_checkpoint.pkl')
    
    # Try to load checkpoint
    checkpoint_data = None
    if Path(checkpoint_file).exists():
        try:
            import pickle
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            if verbose:
                logger.info(f" Loaded checkpoint from: {checkpoint_file}")
                logger.info(f"  Resuming from n={checkpoint_data['last_n'] + 1}")
                logger.info("")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            checkpoint_data = None
    
    if verbose:
        logger.info("="*80)
        logger.info("OPTIMIZED SIMULATED ANNEALING SUBMISSION")
        if checkpoint_data:
            logger.info("(RESUMING FROM CHECKPOINT)")
        logger.info("="*80)
        logger.info("")
    
    # Initialize or restore from checkpoint
    if checkpoint_data:
        all_tree_data = checkpoint_data['all_tree_data']
        side_lengths = checkpoint_data['side_lengths']
        start_n = checkpoint_data['last_n'] + 1
        start_time = time.time() - checkpoint_data['elapsed_time']
    else:
        all_tree_data = []
        side_lengths = []
        start_n = 1
        start_time = time.time()
    
    for n in range(start_n, 201):
        # More iterations for small n
        iters = iterations_small if n <= 20 else iterations_large
        
        if verbose:
            logger.info(f"Optimizing n={n} ({iters} iterations, {trials} trials)...")
        
        trees, score = optimize_with_multiple_trials(
            n, 
            iterations=iters,
            trials=trials,
            verbose=False  # Don't spam logs
        )
        
        side = calculate_bounding_square(trees)
        side_lengths.append(side)
        
        for tree in trees:
            all_tree_data.append([tree.center_x, tree.center_y, tree.angle])
        
        if verbose:
            logger.info(f"  Final score: {score:.6f}")
            
            if n % 5 == 0:
                elapsed = time.time() - start_time
                current_score = calculate_total_score(side_lengths)
                logger.info(f"  Cumulative score: {current_score:.6f}")
                logger.info(f"  Time elapsed: {elapsed/60:.1f} min")
                logger.info(f"  Estimated remaining: {(200-n) * (elapsed/n) / 60:.1f} min")
            logger.info("")
        
        # Save checkpoint every 5 configs
        if n % 5 == 0:
            # try:
            #     import pickle
            #     checkpoint = {
            #         'last_n': n,
            #         'all_tree_data': all_tree_data,
            #         'side_lengths': side_lengths,
            #         'elapsed_time': time.time() - start_time
            #     }
            #     with open(checkpoint_file, 'wb') as f:
            #         pickle.dump(checkpoint, f)
            #     if verbose:
            #         logger.info(f"  Checkpoint saved")
            #         logger.info("")
            # except Exception as e:
            #     logger.warning(f"Failed to save checkpoint: {e}")
            try:
                import pickle
                checkpoint = {
                    'last_n': n,
                    'all_tree_data': all_tree_data,
                    'side_lengths': side_lengths,
                    'elapsed_time': time.time() - start_time
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint, f)
                if verbose:
                    logger.info(f"  Checkpoint saved")
                    
                # Backup to persistent storage
                try:
                    import os
                    
                    # Kaggle
                    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
                        output_dir = Path('/kaggle/working')
                        if output_dir.exists():
                            import shutil
                            backup_path = output_dir / Path(checkpoint_file).name
                            shutil.copy(checkpoint_file, backup_path)
                            logger.info(f"  ✓ Backed up to {backup_path}")
                    
                    # Colab with Google Drive
                    elif 'COLAB_GPU' in os.environ or os.path.exists('/content'):
                        gdrive_path = Path('/content/drive/MyDrive/kaggle_checkpoints')
                        if gdrive_path.exists():
                            import shutil
                            backup_path = gdrive_path / Path(checkpoint_file).name
                            shutil.copy(checkpoint_file, backup_path)
                            logger.info(f"  ✓ Backed up to Google Drive: {backup_path}")
                except:
                    pass
                    
                logger.info("")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
    
    # Export final submission
    export_submission(all_tree_data, output_csv)
    
    total_score = calculate_total_score(side_lengths)
    elapsed = time.time() - start_time
    
    if verbose:
        logger.info("="*80)
        logger.info("FINAL RESULTS")
        logger.info("="*80)
        logger.info(f"Total score: {total_score:.6f}")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"✓ Saved to: {output_csv}")
    
    # Clean up checkpoint
    try:
        Path(checkpoint_file).unlink()
        if verbose:
            logger.info(f"✓ Checkpoint cleaned up")
    except:
        pass
    
    return total_score


def main():
    parser = argparse.ArgumentParser(description='Optimized SA')
    parser.add_argument('--n', type=int, help='Optimize single n')
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--full-submission', action='store_true')
    parser.add_argument('--output', type=str, default='../submissions/sa_submission.csv')
    
    args = parser.parse_args()
    
    if args.full_submission:

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

        
        checkpoint_file = args.output.replace('.csv', '_checkpoint.pkl')
        
        total_score = generate_sa_submission(
            args.output,
            iterations_small=args.iterations,
            iterations_large=max(30000, args.iterations // 2),
            trials=args.trials,
            checkpoint_file=checkpoint_file
        )
        
        scores_file = Path(args.output).parent / 'scores.txt'
        with open(scores_file, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{Path(args.output).name}: {float(total_score):.6f} (optimized_sa) - {timestamp}\n")
        
        logger.info(f"✓ Score logged to: {scores_file}")
        
    elif args.n:
        logger.info(f"Optimizing n={args.n}")
        trees, score = optimize_with_multiple_trials(
            args.n,
            iterations=args.iterations,
            trials=args.trials
        )
        
        logger.info("")
        logger.info("Tree positions:")
        for i, tree in enumerate(trees):
            logger.info(f"  {args.n:03d}_{i},s{float(tree.center_x):.6f},"
                       f"s{float(tree.center_y):.6f},s{float(tree.angle):.1f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()