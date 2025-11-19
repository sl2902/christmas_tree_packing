"""
Optimization strategies for improving tree packing.
"""
from decimal import Decimal
from tree_geometry import ChristmasTree
from utils import calculate_bounding_square, check_collision


# Configuration
DEFAULT_TREES_TO_OPTIMIZE = 5
DEFAULT_ANGLE_STEP = 20


def optimize_rotation_for_tree(tree_idx, placed_trees, angle_step=DEFAULT_ANGLE_STEP):
    """
    Find the best rotation angle for a specific tree.
    
    Args:
        tree_idx: Index of tree to optimize
        placed_trees: List of all placed trees
        angle_step: Angular increment in degrees
    
    Returns:
        Best angle (Decimal)
    """
    tree = placed_trees[tree_idx]
    original_angle = tree.angle
    center_x = tree.center_x
    center_y = tree.center_y
    
    best_angle = original_angle
    best_side_length = calculate_bounding_square(placed_trees)
    
    # Try different angles
    angles_to_try = [Decimal(str(a)) for a in range(0, 360, angle_step)]
    
    for test_angle in angles_to_try:
        # Create test tree with new angle
        test_tree = ChristmasTree(str(center_x), str(center_y), str(test_angle))
        
        # Check for collisions
        if not check_collision(test_tree.polygon, placed_trees, exclude_idx=tree_idx):
            # Temporarily update
            old_tree = placed_trees[tree_idx]
            placed_trees[tree_idx] = test_tree
            
            # Calculate new bounding square
            new_side_length = calculate_bounding_square(placed_trees)
            
            if new_side_length < best_side_length:
                best_side_length = new_side_length
                best_angle = test_angle
            
            # Restore
            placed_trees[tree_idx] = old_tree
    
    return best_angle


def rotation_optimization(placed_trees, num_trees_to_optimize=None, 
                         angle_step=None, max_iterations=3):
    """
    Optimize rotations for recently placed trees.
    
    This is a fast optimization that only adjusts the last N trees.
    
    Args:
        placed_trees: List of placed trees
        num_trees_to_optimize: How many recent trees to optimize (default: adaptive)
        angle_step: Angular increment in degrees (default: adaptive)
        max_iterations: Number of passes through trees
    
    Returns:
        Tuple of (optimized_trees, side_length)
    """
    num_trees = len(placed_trees)
    
    # Adaptive parameters if not specified
    if num_trees_to_optimize is None:
        if num_trees <= 20:
            num_trees_to_optimize = min(num_trees, 5)
        elif num_trees <= 50:
            num_trees_to_optimize = min(num_trees, 10)
        else:
            num_trees_to_optimize = 5
    
    if angle_step is None:
        if num_trees <= 20:
            angle_step = 15
        elif num_trees <= 50:
            angle_step = 20
        else:
            angle_step = 30 
    
    if max_iterations == 3:  # Only if using default
        if num_trees <= 20:
            max_iterations = 2  # Small configs don't need many iterations
        elif num_trees > 100:
            max_iterations = 1  
    
    # Only optimize last N trees
    start_idx = max(0, num_trees - num_trees_to_optimize)
    
    for iteration in range(max_iterations):
        improved = False
        
        for tree_idx in range(start_idx, num_trees):
            old_angle = placed_trees[tree_idx].angle
            best_angle = optimize_rotation_for_tree(tree_idx, placed_trees, angle_step)
            
            if best_angle != old_angle:
                # Update with better angle
                tree = placed_trees[tree_idx]
                placed_trees[tree_idx] = ChristmasTree(
                    str(tree.center_x), 
                    str(tree.center_y), 
                    str(best_angle)
                )
                improved = True
        
        if not improved:
            break
    
    final_side = calculate_bounding_square(placed_trees)
    return placed_trees, final_side


def full_rotation_optimization(placed_trees, angle_step=15, max_iterations=3):
    """
    Optimize rotations for all trees (slower but more thorough).
    
    Args:
        placed_trees: List of placed trees
        angle_step: Angular increment in degrees
        max_iterations: Number of passes through all trees
    
    Returns:
        Tuple of (optimized_trees, side_length)
    """
    num_trees = len(placed_trees)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Optimize each tree
        for tree_idx in range(num_trees):
            old_angle = placed_trees[tree_idx].angle
            best_angle = optimize_rotation_for_tree(tree_idx, placed_trees, angle_step)
            
            if best_angle != old_angle:
                tree = placed_trees[tree_idx]
                placed_trees[tree_idx] = ChristmasTree(
                    str(tree.center_x), 
                    str(tree.center_y), 
                    str(best_angle)
                )
                improved = True
        
        if not improved:
            break
    
    final_side = calculate_bounding_square(placed_trees)
    return placed_trees, final_side