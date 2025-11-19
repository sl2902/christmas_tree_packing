"""
Tree geometry and spatial operations.
"""
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon

# Set precision for Decimal
getcontext().prec = 25
SCALE_FACTOR = Decimal('1e15')


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""
    
    # Tree dimensions (class constants)
    TRUNK_WIDTH = Decimal('0.15')
    TRUNK_HEIGHT = Decimal('0.2')
    BASE_WIDTH = Decimal('0.7')
    MID_WIDTH = Decimal('0.4')
    TOP_WIDTH = Decimal('0.25')
    TIP_Y = Decimal('0.8')
    TIER_1_Y = Decimal('0.5')
    TIER_2_Y = Decimal('0.25')
    BASE_Y = Decimal('0.0')
    
    def __init__(self, center_x='0', center_y='0', angle='0'):
        """
        Initialize a Christmas tree.
        
        Args:
            center_x: X-coordinate of tree base (default '0')
            center_y: Y-coordinate of tree base (default '0')
            angle: Rotation angle in degrees (default '0')
        """
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        
        # Create the tree polygon
        self.polygon = self._create_polygon()
    
    def _create_polygon(self):
        """Create the tree polygon with current position and rotation."""
        trunk_w = self.TRUNK_WIDTH
        trunk_h = self.TRUNK_HEIGHT
        base_w = self.BASE_WIDTH
        mid_w = self.MID_WIDTH
        top_w = self.TOP_WIDTH
        tip_y = self.TIP_Y
        tier_1_y = self.TIER_1_Y
        tier_2_y = self.TIER_2_Y
        base_y = self.BASE_Y
        trunk_bottom_y = -trunk_h
        
        # Create initial polygon (centered at origin, upright)
        initial_polygon = Polygon([
            # Tip
            (Decimal('0.0') * SCALE_FACTOR, tip_y * SCALE_FACTOR),
            # Right side - Top Tier
            (top_w / Decimal('2') * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            (top_w / Decimal('4') * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            # Right side - Middle Tier
            (mid_w / Decimal('2') * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            (mid_w / Decimal('4') * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            # Right side - Bottom Tier
            (base_w / Decimal('2') * SCALE_FACTOR, base_y * SCALE_FACTOR),
            # Right Trunk
            (trunk_w / Decimal('2') * SCALE_FACTOR, base_y * SCALE_FACTOR),
            (trunk_w / Decimal('2') * SCALE_FACTOR, trunk_bottom_y * SCALE_FACTOR),
            # Left Trunk
            (-(trunk_w / Decimal('2')) * SCALE_FACTOR, trunk_bottom_y * SCALE_FACTOR),
            (-(trunk_w / Decimal('2')) * SCALE_FACTOR, base_y * SCALE_FACTOR),
            # Left side - Bottom Tier
            (-(base_w / Decimal('2')) * SCALE_FACTOR, base_y * SCALE_FACTOR),
            # Left side - Middle Tier
            (-(mid_w / Decimal('4')) * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            (-(mid_w / Decimal('2')) * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            # Left side - Top Tier
            (-(top_w / Decimal('4')) * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            (-(top_w / Decimal('2')) * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
        ])
        
        # Rotate around origin
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        
        # Translate to final position
        translated = affinity.translate(
            rotated,
            xoff=float(self.center_x * SCALE_FACTOR),
            yoff=float(self.center_y * SCALE_FACTOR)
        )
        
        return translated
    
    def update_rotation(self, new_angle):
        """
        Update the tree's rotation angle.
        
        Args:
            new_angle: New rotation angle in degrees
        """
        self.angle = Decimal(str(new_angle))
        self.polygon = self._create_polygon()
    
    def update_position(self, new_x, new_y):
        """
        Update the tree's position.
        
        Args:
            new_x: New x-coordinate
            new_y: New y-coordinate
        """
        self.center_x = Decimal(str(new_x))
        self.center_y = Decimal(str(new_y))
        self.polygon = self._create_polygon()
    
    def __repr__(self):
        return f"ChristmasTree(x={self.center_x}, y={self.center_y}, angle={self.angle}°)"