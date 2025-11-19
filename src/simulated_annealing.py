"""
Simulated Annealing optimizer for tree packing.

This module implements SA to optimize both positions and rotations of trees,
allowing escape from local minima through probabilistic acceptance of worse solutions.
"""
import random
import math
from decimal import Decimal
from loguru import logger
from tree_geometry import ChristmasTree
from utils import calculate_bounding_square, check_collision


class SimulatedAnnealing:
    """Simulated Annealing optimizer for tree packing."""
    
    def __init__(self, 
                 initial_temp=1000,
                 cooling_rate=0.995,
                 min_temp=0.1,
                 max_iterations=None):
        """
        Initialize SA parameters.
        
        Args:
            initial_temp: Starting temperature
            cooling_rate: Cooling factor (0-1), lower = slower cooling
            min_temp: Stopping temperature
            max_iterations: Max iterations (overrides temperature stopping)
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
    
    def optimize(self, placed_trees, verbose=False):
        """
        Optimize tree placement using Simulated Annealing.
        
        Args:
            placed_trees: List of ChristmasTree objects
            verbose: Print progress
        
        Returns:
            Tuple of (optimized_trees, final_side_length)
        """
        num_trees = len(placed_trees)
        
        # Current solution
        current = list(placed_trees)
        current_score = float(calculate_bounding_square(current))
        
        # Best solution found
        best = list(current)
        best_score = current_score
        
        # SA parameters
        temp = self.initial_temp
        iteration = 0
        accepts = 0
        rejects = 0
        improvements = 0
        
        if verbose:
            logger.info(f"SA: Starting optimization for {num_trees} trees")
            logger.info(f"    Initial score: {current_score:.4f}")
        
        # Main SA loop
        while temp > self.min_temp:
            if self.max_iterations and iteration >= self.max_iterations:
                break
            
            # Generate neighbor solution
            new_solution = self._perturb(current, temp, num_trees)
            
            if new_solution is None:
                # Perturbation failed (collision), try again
                iteration += 1
                continue
            
            # Evaluate new solution
            new_score = float(calculate_bounding_square(new_solution))
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta < 0:
                # Better solution - always accept
                current = new_solution
                current_score = new_score
                accepts += 1
                improvements += 1
                
                # Update best
                if current_score < best_score:
                    best = list(current)
                    best_score = current_score
            else:
                # Worse solution - accept with probability
                acceptance_prob = math.exp(-delta / temp)
                if random.random() < acceptance_prob:
                    current = new_solution
                    current_score = new_score
                    accepts += 1
                else:
                    rejects += 1
            
            # Cool down
            temp *= self.cooling_rate
            iteration += 1
            
            # Progress report
            if verbose and iteration % 500 == 0:
                accept_rate = accepts / (accepts + rejects) * 100 if (accepts + rejects) > 0 else 0
                logger.info(f"    Iter {iteration}: temp={temp:.2f}, "
                      f"current={current_score:.4f}, best={best_score:.4f}, "
                      f"accept_rate={accept_rate:.1f}%")
        
        if verbose:
            total_attempts = accepts + rejects
            accept_rate = accepts / total_attempts * 100 if total_attempts > 0 else 0
            improvement = (current_score - best_score) / current_score * 100 if current_score > 0 else 0
            logger.info(f"    Final: {current_score:.4f} -> {best_score:.4f} "
                  f"({improvement:.2f}% improvement)")
            logger.info(f"    Stats: {iteration} iterations, {improvements} improvements, "
                  f"{accept_rate:.1f}% acceptance rate")
        
        return best, Decimal(str(best_score))
    
    def _perturb(self, trees, temp, num_trees):
        """
        Create a perturbed solution (neighbor).
        
        Perturbation types:
        1. Rotate a random tree
        2. Move a random tree slightly
        3. Swap two trees (advanced)
        
        Args:
            trees: Current tree configuration
            temp: Current temperature (affects perturbation magnitude)
            num_trees: Number of trees
        
        Returns:
            New tree configuration or None if invalid
        """
        # Copy current solution
        new_trees = list(trees)
        
        # Choose perturbation type
        perturbation_type = random.choice(['rotate', 'move', 'rotate'])  # Favor rotation
        
        # Choose which tree to perturb (favor recently added trees)
        if num_trees <= 5:
            tree_idx = random.randint(0, num_trees - 1)
        else:
            # 70% chance to pick from last 30% of trees
            if random.random() < 0.7:
                start_idx = int(num_trees * 0.7)
                tree_idx = random.randint(start_idx, num_trees - 1)
            else:
                tree_idx = random.randint(0, num_trees - 1)
        
        tree = trees[tree_idx]
        
        if perturbation_type == 'rotate':
            # Rotate tree by small angle (temperature-dependent)
            max_angle_change = 20 * (temp / self.initial_temp)  # Reduced from 30
            angle_delta = random.uniform(-max_angle_change, max_angle_change)
            new_angle = (float(tree.angle) + angle_delta) % 360
            
            new_tree = ChristmasTree(
                str(tree.center_x),
                str(tree.center_y),
                str(new_angle)
            )
        
        elif perturbation_type == 'move':
            # Move tree slightly (temperature-dependent)
            max_move = 0.15 * (temp / self.initial_temp)  # Reduced from 0.3
            dx = random.uniform(-max_move, max_move)
            dy = random.uniform(-max_move, max_move)
            
            new_x = float(tree.center_x) + dx
            new_y = float(tree.center_y) + dy
            
            new_tree = ChristmasTree(
                str(new_x),
                str(new_y),
                str(tree.angle)
            )
        else:
            # Fallback - should not reach here
            return None
        
        # Check for collisions
        new_trees[tree_idx] = new_tree
        if check_collision(new_tree.polygon, new_trees, exclude_idx=tree_idx):
            return None  # Invalid move
        
        return new_trees


def simulated_annealing_optimization(placed_trees, 
                                     initial_temp=None,
                                     cooling_rate=0.995,
                                     max_iterations=None,
                                     verbose=False):
    """
    Convenience function for SA optimization with adaptive parameters.
    
    Args:
        placed_trees: List of ChristmasTree objects
        initial_temp: Starting temperature (adaptive if None)
        cooling_rate: Cooling rate
        max_iterations: Max iterations (adaptive if None)
        verbose: Print progress
    
    Returns:
        Tuple of (optimized_trees, side_length)
    """
    num_trees = len(placed_trees)
    
    # Adaptive parameters based on problem size
    if initial_temp is None:
        if num_trees <= 10:
            initial_temp = 300
        elif num_trees <= 30:
            initial_temp = 500
        elif num_trees <= 100:
            initial_temp = 400
        else:
            initial_temp = 300
    
    if max_iterations is None:
        if num_trees <= 10:
            max_iterations = 3000
        elif num_trees <= 30:
            max_iterations = 5000
        elif num_trees <= 100:
            max_iterations = 4000  # Increased from 2000
        else:
            max_iterations = 2000  # Increased from 1000
    
    # Create and run SA
    sa = SimulatedAnnealing(
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        min_temp=0.1,
        max_iterations=max_iterations
    )
    
    return sa.optimize(placed_trees, verbose=verbose)


def hybrid_optimization(placed_trees, verbose=False):
    """
    Hybrid approach: Rotation optimization followed by SA.
    
    This often gives better results than either method alone.
    
    Args:
        placed_trees: List of ChristmasTree objects
        verbose: Print progress
    
    Returns:
        Tuple of (optimized_trees, side_length)
    """
    from optimization import rotation_optimization
    
    if verbose:
        initial_side = calculate_bounding_square(placed_trees)
        logger.info(f"Hybrid optimization starting: {float(initial_side):.4f}")
    
    # Phase 1: Quick rotation optimization
    trees_rotated, side_rotated = rotation_optimization(placed_trees)
    
    if verbose:
        logger.info(f"  After rotation opt: {float(side_rotated):.4f}")
    
    # Phase 2: SA for fine-tuning
    num_trees = len(trees_rotated)
    if num_trees <= 30:
        sa_iters = 2000
    elif num_trees <= 100:
        sa_iters = 1500
    else:
        sa_iters = 1000
    
    trees_sa, side_sa = simulated_annealing_optimization(
        trees_rotated,
        max_iterations=sa_iters,
        verbose=verbose
    )
    
    if verbose:
        improvement = (initial_side - side_sa) / initial_side * 100
        logger.info(f"  Final: {float(side_sa):.4f} ({float(improvement):.2f}% total improvement)")
    
    return trees_sa, side_sa