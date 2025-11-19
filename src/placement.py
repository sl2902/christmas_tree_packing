"""
Tree placement algorithms.
"""
import math
import random
from decimal import Decimal
from shapely import affinity
from shapely.strtree import STRtree
from tree_geometry import ChristmasTree, SCALE_FACTOR

from utils import calculate_bounding_square


def generate_weighted_angle():
    """
    Generate a random angle with weighted distribution.
    Weights favor corners (abs(sin(2*angle))).
    
    Returns:
        Angle in radians
    """
    while True:
        angle = random.uniform(0, 2 * math.pi)
        if random.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle


def greedy_placement(num_trees, existing_trees=None, num_attempts=10):
    """
    Greedy incremental placement algorithm.
    
    Builds on existing placement by adding new trees one at a time.
    Each tree is placed as close to center as possible without collision.
    
    Args:
        num_trees: Total number of trees to have
        existing_trees: List of already placed trees (optional)
        num_attempts: Number of random angles to try per tree
    
    Returns:
        Tuple of (placed_trees, side_length)
    """
    if num_trees == 0:
        return [], Decimal('0')

    if existing_trees is None:
        placed_trees = []
    else:
        placed_trees = list(existing_trees)

    num_to_add = num_trees - len(placed_trees)

    if num_to_add > 0:
        # Create unplaced trees with random angles
        unplaced_trees = [
            ChristmasTree(angle=random.uniform(0, 360)) 
            for _ in range(num_to_add)
        ]
        
        # First tree goes at origin if starting from scratch
        if not placed_trees:
            placed_trees.append(unplaced_trees.pop(0))

        # Place remaining trees
        for tree_to_place in unplaced_trees:
            placed_polygons = [p.polygon for p in placed_trees]
            tree_index = STRtree(placed_polygons)

            best_px = None
            best_py = None
            min_radius = Decimal('Infinity')

            # Try multiple random directions
            for _ in range(num_attempts):
                # Start at radius 20 from center
                angle = generate_weighted_angle()
                vx = Decimal(str(math.cos(angle)))
                vy = Decimal(str(math.sin(angle)))

                radius = Decimal('20.0')
                step_in = Decimal('0.5')

                # Move inward until collision
                collision_found = False
                while radius >= 0:
                    px = radius * vx
                    py = radius * vy

                    candidate_poly = affinity.translate(
                        tree_to_place.polygon,
                        xoff=float(px * SCALE_FACTOR),
                        yoff=float(py * SCALE_FACTOR)
                    )

                    # Check for collision
                    possible_indices = tree_index.query(candidate_poly)
                    if any((candidate_poly.intersects(placed_polygons[i]) and 
                           not candidate_poly.touches(placed_polygons[i]))
                           for i in possible_indices):
                        collision_found = True
                        break
                    radius -= step_in

                # Back out until no collision
                if collision_found:
                    step_out = Decimal('0.05')
                    while True:
                        radius += step_out
                        px = radius * vx
                        py = radius * vy

                        candidate_poly = affinity.translate(
                            tree_to_place.polygon,
                            xoff=float(px * SCALE_FACTOR),
                            yoff=float(py * SCALE_FACTOR)
                        )

                        possible_indices = tree_index.query(candidate_poly)
                        if not any((candidate_poly.intersects(placed_polygons[i]) and 
                                   not candidate_poly.touches(placed_polygons[i]))
                                   for i in possible_indices):
                            break
                else:
                    # No collision even at center
                    radius = Decimal('0')
                    px = Decimal('0')
                    py = Decimal('0')

                # Keep best position
                if radius < min_radius:
                    min_radius = radius
                    best_px = px
                    best_py = py

            # Place the tree at best position
            tree_to_place.center_x = best_px
            tree_to_place.center_y = best_py
            tree_to_place.polygon = affinity.translate(
                tree_to_place.polygon,
                xoff=float(tree_to_place.center_x * SCALE_FACTOR),
                yoff=float(tree_to_place.center_y * SCALE_FACTOR),
            )
            placed_trees.append(tree_to_place)

    # Calculate bounding square
    side_length = calculate_bounding_square(placed_trees)

    return placed_trees, side_length