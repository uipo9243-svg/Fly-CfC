import math
import numpy as np
import pybullet as p

class AcrobaticArena:
    """
    3D Acrobatic Flight Arena for Fly-CfC v3:
    Features:
      1. Angled Gates (slanted at +-45 degrees) requiring banked turns (Roll).
      2. Multi-level horizontal crossbars requiring pitch dive and pull-up.
      3. Tight chicane slalom gates requiring high-speed body saccades.
      4. Bounded boundary arena with colored obstacles.
    """
    def __init__(self, client_id, arena_size=16.0):
        self.client = client_id
        self.arena_size = arena_size
        self.obstacle_ids = []
        self.gates = []
        
    def build_arena(self, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
            
        self.obstacle_ids.clear()
        self.gates.clear()
        
        # 1. Outer Boundary Walls (16m x 16m x 4m)
        half = self.arena_size / 2.0
        wall_color = [0.25, 0.28, 0.35, 1.0]
        wall_spec = [
            (-half, 0.0, 0.2, self.arena_size),
            ( half, 0.0, 0.2, self.arena_size),
            (0.0, -half, self.arena_size, 0.2),
            (0.0,  half, self.arena_size, 0.2),
        ]
        for wx, wy, l, w in wall_spec:
            c_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l/2, w/2, 2.5], physicsClientId=self.client)
            v_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[l/2, w/2, 2.5], rgbaColor=wall_color, physicsClientId=self.client)
            b_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_id, baseVisualShapeIndex=v_id,
                                     basePosition=[wx, wy, 2.5], physicsClientId=self.client)
            self.obstacle_ids.append(b_id)
            
        # 2. Gate 1: Entrance Chicane - Pair of vertical pillars (y = -4.5m)
        # Wide entrance (3.4m)
        self._add_cylinder(-1.7, -4.5, 0.35, 4.0, [0.15, 0.75, 0.95, 1.0])
        self._add_cylinder( 1.7, -4.5, 0.35, 4.0, [0.95, 0.25, 0.25, 1.0])
        self.gates.append({'center': [0.0, -4.5, 1.2], 'passed': False, 'angle': 0.0})
        
        # 3. Gate 2: Angled Banked Gate (+45 degrees slant) (y = -2.0m)
        # Left post lower, right post higher -> forces right bank (+45 deg roll)
        self._add_slanted_gate(center_x=-0.8, center_y=-2.0, center_z=1.3, slant_deg=40.0, gap=1.8, color=[0.98, 0.82, 0.15, 1.0])
        self.gates.append({'center': [-0.8, -2.0, 1.3], 'passed': False, 'angle': 40.0})
        
        # 4. Multi-level Obstacle: High/Low Crossbar (y = 0.0m)
        # Forces dive to z=0.9m or climb
        self._add_box(0.5, 0.0, 2.4, halfExtents=[1.8, 0.2, 0.6], color=[0.85, 0.3, 0.9, 1.0])
        # Underpass clearance: floor to 1.8m
        self.gates.append({'center': [0.5, 0.0, 1.1], 'passed': False, 'angle': 0.0})
        
        # 5. Gate 3: Angled Banked Gate (-45 degrees slant) (y = 2.2m)
        # Forces left bank (-45 deg roll)
        self._add_slanted_gate(center_x=1.8, center_y=2.2, center_z=1.4, slant_deg=-40.0, gap=1.8, color=[0.20, 0.88, 0.35, 1.0])
        self.gates.append({'center': [1.8, 2.2, 1.4], 'passed': False, 'angle': -40.0})
        
        # 6. Saccade Chicane (y = 4.2m)
        # Forces rapid 90-degree yaw turn to loop back
        self._add_cylinder(-1.2, 4.5, 0.40, 4.0, [0.98, 0.45, 0.15, 1.0])
        self._add_cylinder( 2.2, 4.8, 0.40, 4.0, [0.95, 0.22, 0.22, 1.0])
        self._add_cylinder( 4.5, 2.5, 0.40, 4.0, [0.15, 0.75, 0.95, 1.0])
        self._add_cylinder( 4.2, -1.0, 0.40, 4.0, [0.88, 0.25, 0.88, 1.0])
        self._add_cylinder( 3.5, -4.0, 0.40, 4.0, [0.20, 0.88, 0.35, 1.0])
        
        # Additional scattered slalom pillars
        self._add_cylinder(-3.8, -1.5, 0.40, 4.0, [0.5, 0.6, 0.7, 1.0])
        self._add_cylinder(-3.8,  2.0, 0.40, 4.0, [0.5, 0.6, 0.7, 1.0])
        self._add_cylinder(-1.5,  1.2, 0.35, 4.0, [0.7, 0.5, 0.6, 1.0])

    def _add_cylinder(self, x, y, r, h, color):
        c_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h, physicsClientId=self.client)
        v_id = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=color, physicsClientId=self.client)
        b_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_id, baseVisualShapeIndex=v_id,
                                 basePosition=[x, y, h/2.0], physicsClientId=self.client)
        self.obstacle_ids.append(b_id)
        return b_id

    def _add_box(self, x, y, z, halfExtents, color):
        c_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents, physicsClientId=self.client)
        v_id = p.createVisualShape(p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=color, physicsClientId=self.client)
        b_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_id, baseVisualShapeIndex=v_id,
                                 basePosition=[x, y, z], physicsClientId=self.client)
        self.obstacle_ids.append(b_id)
        return b_id

    def _add_slanted_gate(self, center_x, center_y, center_z, slant_deg, gap=1.8, color=None):
        """
        Creates two pillars angled parallel to each other at slant_deg roll angle.
        gap is the perpendicular passage distance between them.
        """
        if color is None:
            color = [0.95, 0.8, 0.2, 1.0]
            
        rad = math.radians(slant_deg)
        # Normal vector perpendicular to the slant
        # slant rotates around Y (flight direction) -> roll
        dx = math.cos(rad) * (gap / 2.0)
        dz = math.sin(rad) * (gap / 2.0)
        
        quat = p.getQuaternionFromEuler([rad, 0.0, 0.0])
        
        # Post 1 (Left/Lower)
        p1_pos = [center_x - dx, center_y, center_z - dz]
        c1 = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.20, height=2.8, physicsClientId=self.client)
        v1 = p.createVisualShape(p.GEOM_CYLINDER, radius=0.20, length=2.8, rgbaColor=color, physicsClientId=self.client)
        b1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c1, baseVisualShapeIndex=v1,
                               basePosition=p1_pos, baseOrientation=quat, physicsClientId=self.client)
        self.obstacle_ids.append(b1)
        
        # Post 2 (Right/Upper)
        p2_pos = [center_x + dx, center_y, center_z + dz]
        c2 = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.20, height=2.8, physicsClientId=self.client)
        v2 = p.createVisualShape(p.GEOM_CYLINDER, radius=0.20, length=2.8, rgbaColor=color, physicsClientId=self.client)
        b2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c2, baseVisualShapeIndex=v2,
                               basePosition=p2_pos, baseOrientation=quat, physicsClientId=self.client)
        self.obstacle_ids.append(b2)
