"""Build class to pack as many Christmas trees as possible inside a square box"""

import math
import os
import random
from decimal import Decimal, getcontext
import time
from pathlib import Path
from loguru import logger

import numpy as np
import pandas as pd

from shapely import affinity, touches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

getcontext().prec = 25
scale_factor = Decimal('1e15')

# DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent
logger.info(DIR_PATH)

class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))
        

def generate_weighted_angle():
    """
    Generates a random angle with a distribution weighted by abs(sin(2*angle)).
    This helps place more trees in corners, and makes the packing less round.
    """
    while True:
        angle = random.uniform(0, 2 * math.pi)
        if random.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle

def calculate_bounding_square(placed_trees):
    """Calculate the side length of the minimum bounding square."""
    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds
    
    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor
    
    width = maxx - minx
    height = maxy - miny
    side_length = max(width, height)
    
    return side_length

def check_collision(tree_polygon, placed_trees, exclude_idx=None):
    """Check if a tree polygon collides with any placed trees."""
    placed_polygons = [p.polygon for p in placed_trees]
    tree_index = STRtree(placed_polygons)
    
    possible_indices = tree_index.query(tree_polygon)
    for i in possible_indices:
        if exclude_idx is not None and i == exclude_idx:
            continue
        if tree_polygon.intersects(placed_polygons[i]) and not tree_polygon.touches(placed_polygons[i]):
            return True
    return False

def quick_optimize_rotations(placed_trees, num_trees_to_optimize=5, angle_step=30):
    """
    Quick optimization: only optimize the last few trees added.
    This is much faster and still effective.
    """
    num_trees = len(placed_trees)
    
    # Only optimize the most recently added trees
    start_idx = max(0, num_trees - num_trees_to_optimize)
    
    for tree_idx in range(start_idx, num_trees):
        tree = placed_trees[tree_idx]
        center_x = tree.center_x
        center_y = tree.center_y
        original_angle = tree.angle
        
        best_angle = original_angle
        best_side_length = calculate_bounding_square(placed_trees)
        
        # Try different angles
        angles_to_try = [Decimal(str(a)) for a in range(0, 360, angle_step)]
        
        for test_angle in angles_to_try:
            test_tree = ChristmasTree(str(center_x), str(center_y), str(test_angle))
            
            if not check_collision(test_tree.polygon, placed_trees, exclude_idx=tree_idx):
                old_tree = placed_trees[tree_idx]
                placed_trees[tree_idx] = test_tree
                
                new_side_length = calculate_bounding_square(placed_trees)
                
                if new_side_length < best_side_length:
                    best_side_length = new_side_length
                    best_angle = test_angle
                
                placed_trees[tree_idx] = old_tree
        
        # Update with best angle
        if best_angle != original_angle:
            placed_trees[tree_idx] = ChristmasTree(
                str(center_x), 
                str(center_y), 
                str(best_angle)
            )
    
    final_side = calculate_bounding_square(placed_trees)
    return placed_trees, final_side

def initialize_trees_baseline(num_trees, existing_trees=None):
    """Baseline greedy placement algorithm from the sample code."""
    if num_trees == 0:
        return [], Decimal('0')

    if existing_trees is None:
        placed_trees = []
    else:
        placed_trees = list(existing_trees)

    num_to_add = num_trees - len(placed_trees)

    if num_to_add > 0:
        unplaced_trees = [
            ChristmasTree(angle=random.uniform(0, 360)) for _ in range(num_to_add)]
        if not placed_trees:
            placed_trees.append(unplaced_trees.pop(0))

        for tree_to_place in unplaced_trees:
            placed_polygons = [p.polygon for p in placed_trees]
            tree_index = STRtree(placed_polygons)

            best_px = None
            best_py = None
            min_radius = Decimal('Infinity')

            for _ in range(10):
                angle = generate_weighted_angle()
                vx = Decimal(str(math.cos(angle)))
                vy = Decimal(str(math.sin(angle)))

                radius = Decimal('20.0')
                step_in = Decimal('0.5')

                collision_found = False
                while radius >= 0:
                    px = radius * vx
                    py = radius * vy

                    candidate_poly = affinity.translate(
                        tree_to_place.polygon,
                        xoff=float(px * scale_factor),
                        yoff=float(py * scale_factor))

                    possible_indices = tree_index.query(candidate_poly)
                    if any((candidate_poly.intersects(placed_polygons[i]) and not
                            candidate_poly.touches(placed_polygons[i]))
                           for i in possible_indices):
                        collision_found = True
                        break
                    radius -= step_in

                if collision_found:
                    step_out = Decimal('0.05')
                    while True:
                        radius += step_out
                        px = radius * vx
                        py = radius * vy

                        candidate_poly = affinity.translate(
                            tree_to_place.polygon,
                            xoff=float(px * scale_factor),
                            yoff=float(py * scale_factor))

                        possible_indices = tree_index.query(candidate_poly)
                        if not any((candidate_poly.intersects(placed_polygons[i]) and not
                                   candidate_poly.touches(placed_polygons[i]))
                                   for i in possible_indices):
                            break
                else:
                    radius = Decimal('0')
                    px = Decimal('0')
                    py = Decimal('0')

                if radius < min_radius:
                    min_radius = radius
                    best_px = px
                    best_py = py

            tree_to_place.center_x = best_px
            tree_to_place.center_y = best_py
            tree_to_place.polygon = affinity.translate(
                tree_to_place.polygon,
                xoff=float(tree_to_place.center_x * scale_factor),
                yoff=float(tree_to_place.center_y * scale_factor),
            )
            placed_trees.append(tree_to_place)

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds

    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor

    width = maxx - minx
    height = maxy - miny
    side_length = max(width, height)

    return placed_trees, side_length

def calculate_total_score(side_lengths):
    """Calculate the total score from a list of (num_trees, side_length) tuples."""
    total_score = Decimal('0')
    for n, side_length in enumerate(side_lengths, 1):
        group_score = (side_length ** 2) / Decimal(n)
        total_score += group_score
    return total_score

def main():
    random.seed(42)
    
    logger.info("="*80)
    logger.info("FAST ROTATION OPTIMIZATION - Optimizing last N trees only")
    logger.info("="*80)
    logger.info('')
    
    # Generate index for submission
    index = [f'{n:03d}_{t}' for n in range(1, 201) for t in range(n)]
    
    # Storage for results
    side_lengths = []
    all_tree_data = []  # Store data for ALL trees in ALL configurations
    current_placed_trees = []
    start_time = time.time()
    
    for n in range(1, 201):
        # Generate baseline solution (incremental)
        current_placed_trees, baseline_side = initialize_trees_baseline(n, existing_trees=current_placed_trees)
        
        # Quick optimize: only optimize last 3-5 trees
        if n <= 30:
            trees_to_opt = min(n, 5)  # Optimize more trees for small configurations
            angle_step = 20
        else:
            trees_to_opt = 3  # Fewer trees for larger configurations
            angle_step = 30  # Coarser angles for speed
        
        current_placed_trees, optimized_side = quick_optimize_rotations(
            current_placed_trees, 
            num_trees_to_optimize=trees_to_opt,
            angle_step=angle_step
        )
        
        side_lengths.append(optimized_side)
        
        # Store ALL trees for this configuration
        for tree in current_placed_trees:
            all_tree_data.append([tree.center_x, tree.center_y, tree.angle])
        
        # Progress update
        if n % 20 == 0:
            elapsed = time.time() - start_time
            current_score = calculate_total_score(side_lengths)
            
            logger.info(f"Progress: {n}/200 trees ({n/2:.0f}% complete)")
            logger.info(f"  Time elapsed: {elapsed:.1f}s")
            logger.info(f"  Current score: {current_score:.6f}")
            logger.info(f"  Avg time per config: {elapsed/n:.2f}s")
            logger.info('')
    
    # Calculate final score
    final_score = calculate_total_score(side_lengths)
    
    logger.info("="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Final optimized score: {final_score:.6f}")
    logger.info(f"Total time: {time.time() - start_time:.1f}s")
    logger.info('')
    
    # Create submission file
    cols = ['x', 'y', 'deg']
    submission = pd.DataFrame(
        index=index, columns=cols, data=all_tree_data).rename_axis('id')
    
    for col in cols:
        submission[col] = submission[col].astype(float).round(decimals=6)
    
    # Prepend 's' to all values
    for col in submission.columns:
        submission[col] = 's' + submission[col].astype('string')
    
    submission.to_csv(f'{DIR_PATH}/results/optimized_submission.csv')
    logger.info(" Submission saved to: optimized_submission.csv")
    
    # Save side lengths for analysis
    side_lengths_df = pd.DataFrame({
        'num_trees': list(range(1, 201)),
        'side_length': [float(s) for s in side_lengths]
    })
    side_lengths_df.to_csv(f'{DIR_PATH}/results/side_lengths.csv', index=False)
    logger.info(" Side lengths saved to: side_lengths.csv")



if __name__ == "__main__":
    main()