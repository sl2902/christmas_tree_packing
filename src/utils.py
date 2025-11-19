"""
Utility functions for tree packing.
"""
import pandas as pd
from decimal import Decimal
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tree_geometry import SCALE_FACTOR
from loguru import logger


def calculate_bounding_square(placed_trees):
    """
    Calculate the side length of the minimum bounding square.
    
    Args:
        placed_trees: List of ChristmasTree objects
    
    Returns:
        Side length as Decimal
    """
    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds
    
    minx = Decimal(bounds[0]) / SCALE_FACTOR
    miny = Decimal(bounds[1]) / SCALE_FACTOR
    maxx = Decimal(bounds[2]) / SCALE_FACTOR
    maxy = Decimal(bounds[3]) / SCALE_FACTOR
    
    width = maxx - minx
    height = maxy - miny
    side_length = max(width, height)
    
    return side_length


def check_collision(tree_polygon, placed_trees, exclude_idx=None):
    """
    Check if a tree polygon collides with any placed trees.
    
    Args:
        tree_polygon: Shapely Polygon to check
        placed_trees: List of ChristmasTree objects
        exclude_idx: Optional index to exclude from check
    
    Returns:
        True if collision detected, False otherwise
    """
    placed_polygons = [p.polygon for p in placed_trees]
    tree_index = STRtree(placed_polygons)
    
    possible_indices = tree_index.query(tree_polygon)
    for i in possible_indices:
        if exclude_idx is not None and i == exclude_idx:
            continue
        if tree_polygon.intersects(placed_polygons[i]) and not tree_polygon.touches(placed_polygons[i]):
            return True
    return False


def calculate_score(side_length, num_trees):
    """
    Calculate score for a single configuration.
    
    Args:
        side_length: Side length of bounding square
        num_trees: Number of trees in configuration
    
    Returns:
        Score as Decimal
    """
    return (side_length ** 2) / Decimal(num_trees)


def calculate_total_score(side_lengths):
    """
    Calculate total score from list of side lengths.
    
    Args:
        side_lengths: List of side lengths for n=1 to N
    
    Returns:
        Total score as Decimal
    """
    total_score = Decimal('0')
    for n, side_length in enumerate(side_lengths, 1):
        group_score = (side_length ** 2) / Decimal(n)
        total_score += group_score
    return total_score


def export_submission(all_tree_data, filename='submission.csv'):
    """
    Export tree data to CSV submission format.
    
    Args:
        all_tree_data: List of [center_x, center_y, angle] for all trees
        filename: Output filename
    """
    
    # Generate index
    index = [f'{n:03d}_{t}' for n in range(1, 201) for t in range(n)]
    
    # Create DataFrame
    cols = ['x', 'y', 'deg']
    submission = pd.DataFrame(
        index=index, columns=cols, data=all_tree_data
    ).rename_axis('id')
    
    # Round to 6 decimals
    for col in cols:
        submission[col] = submission[col].astype(float).round(decimals=6)
    
    # Prepend 's' to all values
    for col in submission.columns:
        submission[col] = 's' + submission[col].astype('string')
    
    submission.to_csv(filename)
    logger.info(f"Submission saved to: {filename}")