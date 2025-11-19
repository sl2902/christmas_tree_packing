"""
Pattern-based placement strategies for tree packing.

Instead of random greedy placement, use mathematical patterns that are known
to pack efficiently: circular/radial for small n, hexagonal grids for medium n.
"""
import math
from decimal import Decimal
from tree_geometry import ChristmasTree
from utils import calculate_bounding_square, check_collision


def circular_packing(num_trees):
    """
    Place trees in concentric circles (radial pattern).
    
    Best for small n (1-30) where symmetric patterns work well.
    
    Args:
        num_trees: Number of trees to place
    
    Returns:
        Tuple of (placed_trees, side_length)
    """
    if num_trees == 1:
        # Single tree at origin
        trees = [ChristmasTree('0', '0', '0')]
        side = calculate_bounding_square(trees)
        return trees, side
    
    placed_trees = []
    
    # First tree at center
    placed_trees.append(ChristmasTree('0', '0', '0'))
    remaining = num_trees - 1
    
    # Place in concentric circles
    radius = 0.8  # Starting radius
    trees_per_ring = 6  # Start with 6 trees in first ring
    
    while remaining > 0:
        # How many trees in this ring?
        trees_in_ring = min(trees_per_ring, remaining)
        
        # Place trees evenly around circle
        for i in range(trees_in_ring):
            angle_rad = 2 * math.pi * i / trees_in_ring
            x = radius * math.cos(angle_rad)
            y = radius * math.sin(angle_rad)
            
            # Angle the tree toward center (or tangent to circle)
            tree_angle = math.degrees(angle_rad) + 90  # Tangent orientation
            
            tree = ChristmasTree(str(x), str(y), str(tree_angle))
            placed_trees.append(tree)
        
        remaining -= trees_in_ring
        radius += 0.8  # Move to next ring
        trees_per_ring = int(trees_per_ring * 1.5)  # More trees in outer rings
    
    side = calculate_bounding_square(placed_trees)
    return placed_trees, side


def hexagonal_packing(num_trees):
    """
    Place trees in a hexagonal grid pattern.
    
    Best for medium n (20-100) where regular tessellation works well.
    Hexagonal packing is the most efficient 2D packing for circles.
    
    Args:
        num_trees: Number of trees to place
    
    Returns:
        Tuple of (placed_trees, side_length)
    """
    if num_trees == 1:
        trees = [ChristmasTree('0', '0', '0')]
        side = calculate_bounding_square(trees)
        return trees, side
    
    placed_trees = []
    
    # Hexagonal grid parameters
    spacing = 0.85  # Distance between trees
    row_offset = spacing * math.sqrt(3) / 2  # Vertical offset between rows
    
    # Calculate rough grid size needed
    grid_size = int(math.ceil(math.sqrt(num_trees)))
    
    count = 0
    row = 0
    
    while count < num_trees:
        # Alternate row offset for hexagonal pattern
        x_offset = (spacing / 2) if row % 2 == 1 else 0
        
        # Number of trees in this row
        trees_in_row = grid_size if row % 2 == 0 else grid_size - 1
        
        for col in range(trees_in_row):
            if count >= num_trees:
                break
            
            x = col * spacing + x_offset - (grid_size * spacing / 2)
            y = row * row_offset - (grid_size * row_offset / 2)
            
            # Vary angles slightly for better packing
            angle = (row * 30 + col * 15) % 360
            
            tree = ChristmasTree(str(x), str(y), str(angle))
            placed_trees.append(tree)
            count += 1
        
        row += 1
    
    side = calculate_bounding_square(placed_trees)
    return placed_trees, side


def spiral_packing(num_trees):
    """
    Place trees in an Archimedean spiral pattern.
    
    Good for medium-large n where we want gradual density increase.
    
    Args:
        num_trees: Number of trees to place
    
    Returns:
        Tuple of (placed_trees, side_length)
    """
    if num_trees == 1:
        trees = [ChristmasTree('0', '0', '0')]
        side = calculate_bounding_square(trees)
        return trees, side
    
    placed_trees = []
    
    # Spiral parameters
    a = 0.2  # How tight the spiral is
    b = 0.15  # How fast radius grows
    
    for i in range(num_trees):
        # Archimedean spiral: r = a + b*theta
        theta = i * 0.8  # Angular spacing
        radius = a + b * theta
        
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        
        # Orient tree along spiral (tangent)
        angle = math.degrees(theta) + 90
        
        tree = ChristmasTree(str(x), str(y), str(angle))
        placed_trees.append(tree)
    
    side = calculate_bounding_square(placed_trees)
    return placed_trees, side


def adaptive_pattern_placement(num_trees):
    """
    Choose the best pattern based on number of trees.
    
    Args:
        num_trees: Number of trees to place
    
    Returns:
        Tuple of (placed_trees, side_length)
    """
    if num_trees <= 1:
        trees = [ChristmasTree('0', '0', '0')]
        side = calculate_bounding_square(trees)
        return trees, side
    
    elif num_trees <= 20:
        # Small: circular pattern works best
        return circular_packing(num_trees)
    
    elif num_trees <= 80:
        # Medium: hexagonal grid is optimal
        return hexagonal_packing(num_trees)
    
    else:
        # Large: spiral or hexagonal
        return hexagonal_packing(num_trees)


def hybrid_pattern_placement(num_trees, existing_trees=None):
    """
    Hybrid approach: Use pattern for initialization, then greedy for fine-tuning.
    
    This combines the benefits of good initial structure with local optimization.
    
    Args:
        num_trees: Number of trees to place
        existing_trees: Existing trees (for incremental construction)
    
    Returns:
        Tuple of (placed_trees, side_length)
    """
    if existing_trees is None or len(existing_trees) == 0:
        # Start fresh with pattern
        placed_trees, side = adaptive_pattern_placement(num_trees)
    else:
        # Incremental: use pattern for new tree only
        num_existing = len(existing_trees)
        
        if num_existing < num_trees:
            # Get pattern position for the new tree
            pattern_trees, _ = adaptive_pattern_placement(num_trees)
            
            # Copy existing trees
            placed_trees = list(existing_trees)
            
            # Add new tree from pattern
            # Use position from pattern but check for collisions
            new_tree = pattern_trees[num_existing]
            
            # If collision, fall back to greedy placement near pattern position
            if check_collision(new_tree.polygon, placed_trees):
                # Try nearby positions
                base_x = float(new_tree.center_x)
                base_y = float(new_tree.center_y)
                
                found = False
                for dx in [0, 0.1, -0.1, 0.2, -0.2]:
                    for dy in [0, 0.1, -0.1, 0.2, -0.2]:
                        test_tree = ChristmasTree(
                            str(base_x + dx),
                            str(base_y + dy),
                            str(new_tree.angle)
                        )
                        if not check_collision(test_tree.polygon, placed_trees):
                            placed_trees.append(test_tree)
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    # Still collision, use greedy
                    from placement import greedy_placement
                    placed_trees, _ = greedy_placement(num_trees, existing_trees)
            else:
                placed_trees.append(new_tree)
        else:
            placed_trees = existing_trees
    
    side = calculate_bounding_square(placed_trees)
    return placed_trees, side


def test_patterns(num_trees):
    """
    Test all patterns and return the best one.
    
    Args:
        num_trees: Number of trees to test
    
    Returns:
        Tuple of (best_trees, best_side, pattern_name)
    """
    patterns = {
        'circular': circular_packing,
        'hexagonal': hexagonal_packing,
        'spiral': spiral_packing,
    }
    
    best_trees = None
    best_side = Decimal('Infinity')
    best_pattern = None
    
    for name, pattern_func in patterns.items():
        trees, side = pattern_func(num_trees)
        if side < best_side:
            best_side = side
            best_trees = trees
            best_pattern = name
    
    return best_trees, best_side, best_pattern