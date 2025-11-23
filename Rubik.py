import streamlit as st
import numpy as np
import plotly.graph_objects as go
import random
import time
from rubik.solve import Solver
from rubik.optimize import optimize_moves
from rubik.cube import Cube
import base64


if "scramble_moves" not in st.session_state:
    st.session_state.scramble_moves = 20

class RubiksCube:
    def __init__(self):
        # Corrected face definitions with proper orientation
        self.faces = {
            'U': np.full((3, 3), 'white', dtype='<U6'),    # Up
            'R': np.full((3, 3), 'red', dtype='<U6'),      # Right  <-- fixed typo 're09d' -> 'red'
            'F': np.full((3, 3), 'green', dtype='<U6'),    # Front
            'D': np.full((3, 3), 'yellow', dtype='<U6'),   # Down
            'L': np.full((3, 3), 'orange', dtype='<U6'),   # Left
            'B': np.full((3, 3), 'blue', dtype='<U6')      # Back
        }
        self.color_map = {
            'green': '#00FF00',
            'red': '#FF0000',
            'blue': '#0000FF',
            'orange': '#FFA500',
            'white': '#FFFFFF',
            'yellow': '#FFFF00'
        }

    def is_solved(self):
        for face in self.faces.values():
            if not np.all(face == face[0, 0]):
                return False
        return True

    def rotate_face(self, face_char, clockwise=True):
        """Rotate a face and its adjacent edges"""
        if clockwise:
            # Rotate the face itself clockwise
            self.faces[face_char] = np.rot90(self.faces[face_char], 3)
            # Rotate the adjacent edges
            self._rotate_adjacent_edges(face_char, clockwise)
        else:
            # Rotate the face itself counter-clockwise
            self.faces[face_char] = np.rot90(self.faces[face_char], 1)
            # Rotate the adjacent edges
            self._rotate_adjacent_edges(face_char, clockwise)

    def _rotate_adjacent_edges(self, face_char, clockwise):
        """Rotate the adjacent edge pieces of a face"""
        if face_char == 'F':
            if clockwise:
                # Save top row of U
                temp = self.faces['U'][2, :].copy()
                # U bottom gets L right column reversed
                self.faces['U'][2, :] = self.faces['L'][::-1, 2]
                # L right gets D top row
                self.faces['L'][:, 2] = self.faces['D'][0, :]
                # D top gets R left column reversed
                self.faces['D'][0, :] = self.faces['R'][::-1, 0]
                # R left gets saved U bottom
                self.faces['R'][:, 0] = temp
            else:
                # Counter-clockwise
                temp = self.faces['U'][2, :].copy()
                self.faces['U'][2, :] = self.faces['R'][:, 0]
                self.faces['R'][:, 0] = self.faces['D'][0, ::-1]
                self.faces['D'][0, :] = self.faces['L'][:, 2]
                self.faces['L'][:, 2] = temp[::-1]
                
        elif face_char == 'B':
            if clockwise:
                temp = self.faces['U'][0, :].copy()
                self.faces['U'][0, :] = self.faces['R'][:, 2]
                self.faces['R'][:, 2] = self.faces['D'][2, ::-1]
                self.faces['D'][2, :] = self.faces['L'][:, 0]
                self.faces['L'][:, 0] = temp[::-1]
            else:
                temp = self.faces['U'][0, :].copy()
                self.faces['U'][0, :] = self.faces['L'][::-1, 0]
                self.faces['L'][:, 0] = self.faces['D'][2, :]
                self.faces['D'][2, :] = self.faces['R'][::-1, 2]
                self.faces['R'][:, 2] = temp
                
        elif face_char == 'R':
            if clockwise:
                temp = self.faces['U'][:, 2].copy()
                self.faces['U'][:, 2] = self.faces['F'][:, 2]
                self.faces['F'][:, 2] = self.faces['D'][:, 2]
                self.faces['D'][:, 2] = self.faces['B'][::-1, 0]
                self.faces['B'][:, 0] = temp[::-1]
            else:
                temp = self.faces['U'][:, 2].copy()
                self.faces['U'][:, 2] = self.faces['B'][::-1, 0]
                self.faces['B'][:, 0] = self.faces['D'][::-1, 2]
                self.faces['D'][:, 2] = self.faces['F'][:, 2]
                self.faces['F'][:, 2] = temp
                
        elif face_char == 'L':
            if clockwise:
                temp = self.faces['U'][:, 0].copy()
                self.faces['U'][:, 0] = self.faces['B'][::-1, 2]
                self.faces['B'][:, 2] = self.faces['D'][::-1, 0]
                self.faces['D'][:, 0] = self.faces['F'][:, 0]
                self.faces['F'][:, 0] = temp
            else:
                temp = self.faces['U'][:, 0].copy()
                self.faces['U'][:, 0] = self.faces['F'][:, 0]
                self.faces['F'][:, 0] = self.faces['D'][:, 0]
                self.faces['D'][:, 0] = self.faces['B'][::-1, 2]
                self.faces['B'][:, 2] = temp[::-1]
                
        elif face_char == 'U':
            if clockwise:
                temp = self.faces['F'][0, :].copy()
                self.faces['F'][0, :] = self.faces['R'][0, :]
                self.faces['R'][0, :] = self.faces['B'][0, :]
                self.faces['B'][0, :] = self.faces['L'][0, :]
                self.faces['L'][0, :] = temp
            else:
                temp = self.faces['F'][0, :].copy()
                self.faces['F'][0, :] = self.faces['L'][0, :]
                self.faces['L'][0, :] = self.faces['B'][0, :]
                self.faces['B'][0, :] = self.faces['R'][0, :]
                self.faces['R'][0, :] = temp
                
        elif face_char == 'D':
            if clockwise:
                temp = self.faces['F'][2, :].copy()
                self.faces['F'][2, :] = self.faces['L'][2, :]
                self.faces['L'][2, :] = self.faces['B'][2, :]
                self.faces['B'][2, :] = self.faces['R'][2, :]
                self.faces['R'][2, :] = temp
            else:
                temp = self.faces['F'][2, :].copy()
                self.faces['F'][2, :] = self.faces['R'][2, :]
                self.faces['R'][2, :] = self.faces['B'][2, :]
                self.faces['B'][2, :] = self.faces['L'][2, :]
                self.faces['L'][2, :] = temp

    def apply_move(self, move):
        if move.endswith("'"):
            face_char = move[0]
            clockwise = False
        elif move.endswith("2"):
            face_char = move[0]
            self.apply_move(face_char)
            self.apply_move(face_char)
            return
        else:
            face_char = move
            clockwise = True

        if face_char in self.faces:
            self.rotate_face(face_char, clockwise)
        else:
            raise ValueError(f"Invalid move: {move}")

    def scramble(self, moves=st.session_state.scramble_moves):
        moves_list = []
        possible_moves = ['F', 'R', 'U', 'B', 'L', 'D',
                          "F'", "R'", "U'", "B'", "L'", "D'",
                          'F2', 'R2', 'U2', 'B2', 'L2', 'D2']
        for _ in range(moves):
            m = random.choice(possible_moves)
            self.apply_move(m)
            moves_list.append(m)
        return moves_list

    def get_cube_state(self):
        return {k: v.copy() for k, v in self.faces.items()}
    
    def to_rubik_lib_cube(self):
        """Convert our cube representation to the rubik library's cube format"""
        cube_str = ""
        
        # Up face (white) - positions 0-8
        for i in range(3):
            for j in range(3):
                cube_str += self._color_to_char(self.faces['U'][i, j])
        
        # Right face (red) - positions 9-17
        for i in range(3):
            for j in range(3):
                cube_str += self._color_to_char(self.faces['R'][i, j])
        
        # Front face (green) - positions 18-26
        for i in range(3):
            for j in range(3):
                cube_str += self._color_to_char(self.faces['F'][i, j])
        
        # Down face (yellow) - positions 27-35
        for i in range(3):
            for j in range(3):
                cube_str += self._color_to_char(self.faces['D'][i, j])
        
        # Left face (orange) - positions 36-44
        for i in range(3):
            for j in range(3):
                cube_str += self._color_to_char(self.faces['L'][i, j])
        
        # Back face (blue) - positions 45-53
        for i in range(3):
            for j in range(3):
                cube_str += self._color_to_char(self.faces['B'][i, j])
                
        return Cube(cube_str)
    
    def _color_to_char(self, color):
        """Convert color name to rubik library character"""
        color_map = {
            'white': 'U',
            'red': 'R',
            'green': 'F',
            'yellow': 'D',
            'orange': 'L',
            'blue': 'B'
        }
        return color_map.get(color, 'U')
    
    def _char_to_color(self, char):
        """Convert rubik library character to color name"""
        char_map = {
            'U': 'white',
            'R': 'red',
            'F': 'green',
            'D': 'yellow',
            'L': 'orange',
            'B': 'blue'
        }
        return char_map.get(char, 'white')
    
    def from_rubik_lib_cube(self, rubik_cube):
        """Update our cube from a rubik library cube"""
        cube_str = rubik_cube.flat_str()
        
        # Extract faces from the 54-character string
        idx = 0
        
        # Up face (positions 0-8)
        for i in range(3):
            for j in range(3):
                self.faces['U'][i, j] = self._char_to_color(cube_str[idx])
                idx += 1
        
        # Right face (positions 9-17)
        for i in range(3):
            for j in range(3):
                self.faces['R'][i, j] = self._char_to_color(cube_str[idx])
                idx += 1
        
        # Front face (positions 18-26)
        for i in range(3):
            for j in range(3):
                self.faces['F'][i, j] = self._char_to_color(cube_str[idx])
                idx += 1
        
        # Down face (positions 27-35)
        for i in range(3):
            for j in range(3):
                self.faces['D'][i, j] = self._char_to_color(cube_str[idx])
                idx += 1
        
        # Left face (positions 36-44)
        for i in range(3):
            for j in range(3):
                self.faces['L'][i, j] = self._char_to_color(cube_str[idx])
                idx += 1
        
        # Back face (positions 45-53)
        for i in range(3):
            for j in range(3):
                self.faces['B'][i, j] = self._char_to_color(cube_str[idx])
                idx += 1

    def clone(self):
        """Return a deep copy of this cube for search/apply without mutating original."""
        new = RubiksCube()
        for k, v in self.faces.items():
            new.faces[k] = v.copy()
        return new


class InternalSolver:
	"""Simple IDA* solver using RubiksCube.clone() and apply_move.
	   Heuristic: count of misplaced stickers (divided so heuristic is admissible-ish).
	"""
	def __init__(self, max_depth=20, time_limit=8.0):
		self.max_depth = max_depth
		self.time_limit = time_limit
		self.start_time = None
		self.moves = ['F','R','U','B','L','D',
					  "F'", "R'", "U'", "B'", "L'", "D'",
					  'F2', 'R2', 'U2', 'B2', 'L2', 'D2']

	def _heuristic(self, cube: RubiksCube):
		# Count stickers not equal to face center; each move can fix multiple stickers,
		# divide by 8 to keep heuristic small (admissible-ish for shallow searches).
		cnt = 0
		for face_char, face in cube.faces.items():
			center = face[1,1]
			cnt += np.sum(face != center)
		return int(np.ceil(cnt / 8))

	def _inverse(self, m):
		if m.endswith("'"):
			return m[:-1]
		if m.endswith("2"):
			return m
		return m + "'"

	def solve(self, cube: RubiksCube):
		self.start_time = time.time()
		if cube.is_solved():
			return []
		# IDA* loop
		bound = self._heuristic(cube)
		if bound == 0:
			return []
		path = []

		def dfs(node: RubiksCube, g: int, bound: int, last_move: str):
			if time.time() - self.start_time > self.time_limit:
				return None, float('inf')
			h = self._heuristic(node)
			f = g + h
			if f > bound or g > self.max_depth:
				return None, f
			if node.is_solved():
				return [], f
			min_t = float('inf')
			for mv in self.moves:
				# prune immediate inverse
				if last_move and self._inverse(last_move) == mv:
					continue
				# apply move on clone
				new_node = node.clone()
				new_node.apply_move(mv)
				res, t = dfs(new_node, g+1, bound, mv)
				if res is not None:
					return [mv] + res, t
				if t < min_t:
					min_t = t
			return None, min_t

		while True:
			if time.time() - self.start_time > self.time_limit or bound > self.max_depth:
				return []
			result, t = dfs(cube.clone(), 0, bound, None)
			if result is not None:
				return result
			if t == float('inf'):
				return []
			bound = int(t)


def create_3d_cube_visualization(cube_state, color_map, animation_phase=0, rotating_face=None, clockwise=True):
    fig = go.Figure()
    
    # face normals from previous setup (keeps sticker orientation logic)
    face_positions = {
        'F': {'center': [0, 0, 1.5], 'normal': [0, 0, 1]},
        'R': {'center': [1.5, 0, 0], 'normal': [1, 0, 0]},
        'B': {'center': [0, 0, -1.5], 'normal': [0, 0, -1]},
        'L': {'center': [-1.5, 0, 0], 'normal': [-1, 0, 0]},
        'U': {'center': [0, 1.5, 0], 'normal': [0, 1, 0]},
        'D': {'center': [0, -1.5, 0], 'normal': [0, -1, 0]}
    }
    
    # helper to snap a coordinate to nearest cubie center (layer centers are at -1.5, 0, 1.5)
    layer_centers = [-1.5, 0.0, 1.5]
    def nearest_center(v):
        return min(layer_centers, key=lambda c: abs(c - v))
    
    size = 0.9  # sticker quad size (keeps same sticker scale)
    
    # collect quads per cubie key (cx, cy, cz)
    cubie_quads = {}  # key -> list of (verts[4], color_hex)
    
    # build stickers (same positions as before) but group them by cubie center
    for face_char, face_data in cube_state.items():
        face_info = face_positions[face_char]
        for i in range(3):
            for j in range(3):
                # sticker center as before
                x = j - 1
                y = 1 - i
                if face_char in ['F', 'B', 'L', 'R']:
                    if face_char == 'F':
                        center = [x, y, 1.5]
                    elif face_char == 'B':
                        center = [-x, y, -1.5]
                    elif face_char == 'R':
                        center = [1.5, y, -x]
                    elif face_char == 'L':
                        center = [-1.5, y, x]
                else:
                    if face_char == 'U':
                        center = [x, 1.5, -y]
                    else:  # D
                        center = [x, -1.5, y]

                normal = face_info['normal']
                # compute quad verts (exact same geometry as before)
                if abs(normal[0]) == 1:
                    v0 = [center[0], center[1] - size/2, center[2] - size/2]
                    v1 = [center[0], center[1] + size/2, center[2] - size/2]
                    v2 = [center[0], center[1] + size/2, center[2] + size/2]
                    v3 = [center[0], center[1] - size/2, center[2] + size/2]
                elif abs(normal[1]) == 1:
                    v0 = [center[0] - size/2, center[1], center[2] - size/2]
                    v1 = [center[0] + size/2, center[1], center[2] - size/2]
                    v2 = [center[0] + size/2, center[1], center[2] + size/2]
                    v3 = [center[0] - size/2, center[1], center[2] + size/2]
                else:
                    v0 = [center[0] - size/2, center[1] - size/2, center[2]]
                    v1 = [center[0] + size/2, center[1] - size/2, center[2]]
                    v2 = [center[0] + size/2, center[1] + size/2, center[2]]
                    v3 = [center[0] - size/2, center[1] + size/2, center[2]]

                verts = [v0, v1, v2, v3]
                # determine which cubie these sticker verts belong to by nearest cubie-center
                cx = nearest_center(center[0])
                cy = nearest_center(center[1])
                cz = nearest_center(center[2])
                key = (cx, cy, cz)
                color_hex = color_map.get(face_data[i, j], "#000000")
                cubie_quads.setdefault(key, []).append((verts, color_hex))

    # If a rotating layer is active, compute rotation angle
    angle = 0.0
    if rotating_face is not None and animation_phase > 0:
        angle = animation_phase * 90 * (-1 if clockwise else 1)

    # For each cubie, optionally rotate all its vertices together and add one Mesh3d trace
    for cubie_key, quads in cubie_quads.items():
        cx, cy, cz = cubie_key
        verts_x = []
        verts_y = []
        verts_z = []
        tri_i = []
        tri_j = []
        tri_k = []
        vertex_colors = []
        vc = 0

        # decide if this cubie should be rotated as part of the active rotating face
        rotate_this = False
        if rotating_face == 'F' and abs(cz - 1.5) < 1e-6:
            rotate_this = True
        if rotating_face == 'B' and abs(cz + 1.5) < 1e-6:
            rotate_this = True
        if rotating_face == 'R' and abs(cx - 1.5) < 1e-6:
            rotate_this = True
        if rotating_face == 'L' and abs(cx + 1.5) < 1e-6:
            rotate_this = True
        if rotating_face == 'U' and abs(cy - 1.5) < 1e-6:
            rotate_this = True
        if rotating_face == 'D' and abs(cy + 1.5) < 1e-6:
            rotate_this = True

        for verts, col in quads:
            # rotate quad verts together if cubie is in rotating layer
            transformed = []
            for (vx, vy, vz) in verts:
                if rotate_this and angle != 0.0:
                    vx2, vy2, vz2 = _rotate_sticker(vx, vy, vz, rotating_face, angle)
                else:
                    vx2, vy2, vz2 = vx, vy, vz
                verts_x.append(vx2)
                verts_y.append(vy2)
                verts_z.append(vz2)
                vertex_colors.append(col)
                transformed.append((vx2, vy2, vz2))
            # two triangles for the quad
            tri_i.append(vc + 0); tri_j.append(vc + 1); tri_k.append(vc + 2)
            tri_i.append(vc + 0); tri_j.append(vc + 2); tri_k.append(vc + 3)
            vc += 4

        if vc == 0:
            continue

        fig.add_trace(go.Mesh3d(
            x=verts_x,
            y=verts_y,
            z=verts_z,
            i=tri_i,
            j=tri_j,
            k=tri_k,
            vertexcolor=vertex_colors,
            flatshading=True,
            opacity=1.0,
            showscale=False,
            lighting=dict(ambient=0.7, diffuse=0.6, roughness=0.9, specular=0.05)
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-2.0, 2.0]),
            yaxis=dict(visible=False, range=[-2.0, 2.0]),
            zaxis=dict(visible=False, range=[-2.0, 2.0]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision='constant'
    )
    return fig

def _is_sticker_on_face(face_char, i, j, rotating_face):
    """Check if a sticker is part of the rotating face layer"""
    if rotating_face == 'F':
        return face_char == 'F' or (face_char == 'U' and i == 2) or (face_char == 'D' and i == 0) or (face_char == 'R' and j == 0) or (face_char == 'L' and j == 2)
    elif rotating_face == 'B':
        return face_char == 'B' or (face_char == 'U' and i == 0) or (face_char == 'D' and i == 2) or (face_char == 'R' and j == 2) or (face_char == 'L' and j == 0)
    elif rotating_face == 'R':
        return face_char == 'R' or (face_char == 'U' and j == 2) or (face_char == 'D' and j == 2) or (face_char == 'F' and j == 2) or (face_char == 'B' and j == 0)
    elif rotating_face == 'L':
        return face_char == 'L' or (face_char == 'U' and j == 0) or (face_char == 'D' and j == 0) or (face_char == 'F' and j == 0) or (face_char == 'B' and j == 2)
    elif rotating_face == 'U':
        return face_char == 'U' or (face_char == 'F' and i == 0) or (face_char == 'R' and i == 0) or (face_char == 'B' and i == 0) or (face_char == 'L' and i == 0)
    elif rotating_face == 'D':
        return face_char == 'D' or (face_char == 'F' and i == 2) or (face_char == 'R' and i == 2) or (face_char == 'B' and i == 2) or (face_char == 'L' and i == 2)
    return False

def _rotate_sticker(x, y, z, face, angle):
    """Rotate a sticker around the appropriate axis"""
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    if face == 'F':  # Rotate around Z axis
        center_x, center_y, center_z = 0, 0, 1.5
        x_rot = center_x + (x - center_x) * cos_a - (y - center_y) * sin_a
        y_rot = center_y + (x - center_x) * sin_a + (y - center_y) * cos_a
        return [x_rot, y_rot, z]
    elif face == 'B':  # Rotate around Z axis
        center_x, center_y, center_z = 0, 0, -1.5
        x_rot = center_x + (x - center_x) * cos_a + (y - center_y) * sin_a
        y_rot = center_y - (x - center_x) * sin_a + (y - center_y) * cos_a
        return [x_rot, y_rot, z]
    elif face == 'R':  # Rotate around X axis
        center_x, center_y, center_z = 1.5, 0, 0
        y_rot = center_y + (y - center_y) * cos_a - (z - center_z) * sin_a
        z_rot = center_z + (y - center_y) * sin_a + (z - center_z) * cos_a
        return [x, y_rot, z_rot]
    elif face == 'L':  # Rotate around X axis
        center_x, center_y, center_z = -1.5, 0, 0
        y_rot = center_y + (y - center_y) * cos_a + (z - center_z) * sin_a
        z_rot = center_z - (y - center_y) * sin_a + (z - center_z) * cos_a
        return [x, y_rot, z_rot]
    elif face == 'U':  # Rotate around Y axis
        center_x, center_y, center_z = 0, 1.5, 0
        x_rot = center_x + (x - center_x) * cos_a + (z - center_z) * sin_a
        z_rot = center_z - (x - center_x) * sin_a + (z - center_z) * cos_a
        return [x_rot, y, z_rot]
    elif face == 'D':  # Rotate around Y axis
        center_x, center_y, center_z = 0, -1.5, 0
        x_rot = center_x + (x - center_x) * cos_a - (z - center_z) * sin_a
        z_rot = center_z + (x - center_x) * sin_a + (z - center_z) * cos_a
        return [x_rot, y, z_rot]
    
    return [x, y, z]

def add_sticker(fig, center, normal, color):
    x, y, z = center
    size = 0.9
    if abs(normal[0]) == 1:
        verts = [
            [x, y-size/2, z-size/2], [x, y+size/2, z-size/2],
            [x, y+size/2, z+size/2], [x, y-size/2, z+size/2]
        ]
    elif abs(normal[1]) == 1:
        verts = [
            [x-size/2, y, z-size/2], [x+size/2, y, z-size/2],
            [x+size/2, y, z+size/2], [x-size/2, y, z+size/2]
        ]
    else:
        verts = [
            [x-size/2, y-size/2, z], [x+size/2, y-size/2, z],
            [x+size/2, y+size/2, z], [x-size/2, y+size/2, z]
        ]
    fig.add_trace(go.Mesh3d(
        x=[v[0] for v in verts],
        y=[v[1] for v in verts],
        z=[v[2] for v in verts],
        i=[0,0], j=[1,2], k=[2,3],
        color=color,
        opacity=1,
        flatshading=True
    ))

def create_2d_net_visualization(cube_state, color_map, animation_phase=0, rotating_face=None, clockwise=True):
    fig = go.Figure()
    
    # Define fixed positions for each face in the net
    face_positions = {
        'U': {'x_range': [4, 7], 'y_range': [7, 10]},  # Top center
        'L': {'x_range': [1, 4], 'y_range': [4, 7]},   # Left middle
        'F': {'x_range': [4, 7], 'y_range': [4, 7]},   # Center middle  
        'R': {'x_range': [7, 10], 'y_range': [4, 7]},  # Right middle
        'B': {'x_range': [10, 13], 'y_range': [4, 7]}, # Far right middle
        'D': {'x_range': [4, 7], 'y_range': [1, 4]}    # Bottom center
    }
    
    for face_char, pos_info in face_positions.items():
        face_data = cube_state[face_char]
        x_start, x_end = pos_info['x_range']
        y_start, y_end = pos_info['y_range']
        
        face_width = x_end - x_start
        face_height = y_end - y_start
        sticker_size_x = face_width / 3
        sticker_size_y = face_height / 3
        
        for i in range(3):
            for j in range(3):
                # Calculate sticker position
                x0 = x_start + j * sticker_size_x
                y0 = y_start + (2 - i) * sticker_size_y
                x1 = x0 + sticker_size_x
                y1 = y0 + sticker_size_y
                
                color = color_map.get(face_data[i, j], "#000000")
                
                # Add the sticker as a filled rectangle
                fig.add_trace(go.Scatter(
                    x=[x0, x1, x1, x0, x0],
                    y=[y0, y0, y1, y1, y0],
                    fill="toself",
                    fillcolor=color,
                    line=dict(color="#000000", width=2),
                    mode="lines",
                    showlegend=False,
                    hoverinfo="none"
                ))
    
    # FIXED: Complete layout lockdown to prevent any zooming/shifting
    fig.update_layout(

        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            showline=False,
            range=[0, 14],
            scaleanchor="y",
            scaleratio=1,
            constrain="domain",
            fixedrange=True,
            autorange=False
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            showline=False,
            range=[0, 11],
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            fixedrange=True,
            autorange=False
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        autosize=False,
        # Additional constraints to prevent any interaction
        dragmode=False,
        hovermode=False,
        # Lock the entire layout
        uirevision='constant'  # This is key - prevents any view changes
    )
    
    # Add invisible points at the corners to enforce the fixed range
    fig.add_trace(go.Scatter(
        x=[0, 14, 14, 0],
        y=[0, 0, 11, 11],
        mode='markers',
        marker=dict(size=0, opacity=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    return fig

class RubiksSolver:
    def __init__(self):
        pass

    def solve(self, cube: RubiksCube):
        """
        Use the rubik library solver to solve the cube
        """
        try:
            # Convert our cube to the rubik library format
            rubik_cube = cube.to_rubik_lib_cube()
            
            # Verify the cube is valid
            if not self._is_valid_cube(rubik_cube):
                # try internal solver as fallback when rubik lib says invalid
                internal = InternalSolver(max_depth=20, time_limit=6.0)
                sol = internal.solve(cube)
                return sol
            
            # Solve using the library
            solver = Solver(rubik_cube)
            solver.solve()
            
            # Check if solution was found
            if not solver.moves:
                # try internal solver before giving up
                internal = InternalSolver(max_depth=20, time_limit=6.0)
                sol = internal.solve(cube)
                if sol:
                    return sol
                st.warning("No solution found by library; using fallback")
                return []
            
            # Optimize the moves
            optimized_moves = optimize_moves(solver.moves)
            
            return optimized_moves
            
        except Exception as e:
            # attempt internal solver before using reverse-only fallback
            internal = InternalSolver(max_depth=20, time_limit=8.0)
            sol = internal.solve(cube)
            if sol:
                return sol
            return self._fallback_solution(cube)
    
    def _is_valid_cube(self, rubik_cube):
        """Check if the cube state is valid"""
        try:
            # Try to access cube properties to check validity
            cube_str = rubik_cube.flat_str()
            return len(cube_str) == 54 and all(c in 'URFDLB' for c in cube_str)
        except:
            return False
    
    def _fallback_solution(self, cube):
        """Provide a fallback solution when the main solver fails"""
        # This is a simple approach - just reverse the last moves
        if hasattr(st.session_state, 'move_history') and st.session_state.move_history:
            # Reverse the moves and their directions
            reversed_moves = []
            for move in reversed(st.session_state.move_history[-st.session_state.scramble_moves:]):  # Last 20 moves
                if move.endswith("'"):
                    reversed_moves.append(move[0])  # Remove prime
                elif not move.endswith("2"):
                    reversed_moves.append(move + "'")  # Add prime
                else:
                    reversed_moves.append(move)  # Keep double moves as is
            return reversed_moves
        return []


def main():
    st.set_page_config(page_title="rubik's cube solver", layout="wide")
    # Load the CSS files
    def load_css(filename):
        with open(filename) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # remove the header of main page
    load_css(r"Styles/header_st.css")

    # reduce whitespace in sidebar and main page
    load_css(r"Styles/whitespace_st.css")

    # load border style (for logo)
    load_css(r"Styles/logo_sidebar.css")

    

    if 'cube' not in st.session_state:
        st.session_state.cube = RubiksCube()
        st.session_state.solution = []
        st.session_state.current_step = 0
        st.session_state.auto_playing = False
        st.session_state.move_history = []
        # animation pipeline: progress [0..1], pending_move applied when progress reaches 1
        st.session_state.animation_progress = 0.0
        st.session_state.pending_move = None
        st.session_state.rotating_face = None
        st.session_state.clockwise = True
        # number of internal frames to use for smoothness
        st.session_state.animation_frames = 10
        # user-facing speed multiplier (1..20). higher -> faster progress per rerun
        st.session_state.animation_speed = 30

    # Sidebar with all controls and information
    with st.sidebar:
        # Logo / Title area (kept compact & professional)
        with open("logo.png", "rb") as img_file:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode()

        # Remove top whitespace: set margin-top to -10px and padding-top to 0
        st.markdown(
            f"""
            <div style="text-align:center; margin-top:-10px; padding-top:0;">
                <img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:64px; object-fit:contain;"/>
                <p style="margin:2px 0 8px 0; color:#6c757d; font-size:0.9rem;">
                    Interactive Rubik's Cube Solver
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Recent moves (compact)
        if st.session_state.move_history:
            st.subheader("Recent moves")
            st.write(" → ".join(st.session_state.move_history[-50:]))

        st.markdown("---")

        # Animation settings group
        st.subheader("Animation settings")
        st.session_state.scramble_moves = st.slider(
            "Scramble length",
            min_value=5,
            max_value=100,
            value=st.session_state.scramble_moves,
            step=1,
            help="Number of random moves used when scrambling the cube.",
        )
        st.session_state.animation_frames = st.slider(
            "Smoothness (frames)",
            min_value=4,
            max_value=60,
            value=st.session_state.animation_frames,
            step=1,
            help="Higher = smoother rotation but more CPU.",
        )
        st.session_state.animation_speed = st.slider(
            "Animation speed",
            min_value=1,
            max_value=100,
            value=st.session_state.animation_speed,
            step=1,
            help="Higher = faster animations (shorter pauses).",
        )

        st.markdown("---")

        # Cube actions
        st.subheader("Cube actions")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Scramble", use_container_width=True):
                moves = st.session_state.cube.scramble(st.session_state.scramble_moves)
                st.session_state.move_history = moves.copy()
                st.session_state.solution = []
                st.session_state.current_step = 0
                st.session_state.auto_playing = False
                st.session_state.animation_progress = 0
                st.rerun()
        with c2:
            if st.button("Solve", use_container_width=True):
                with st.spinner("Computing solution..."):
                    solver = RubiksSolver()
                    solution = solver.solve(st.session_state.cube)
                    if solution:
                        st.session_state.solution = solution
                        st.session_state.current_step = 0
                        st.session_state.auto_playing = True
                    else:
                        st.session_state.solution = []
                        st.session_state.auto_playing = False

        if st.button("Reset cube", use_container_width=True):
            st.session_state.cube = RubiksCube()
            st.session_state.solution = []
            st.session_state.current_step = 0
            st.session_state.auto_playing = False
            st.session_state.move_history = []
            st.session_state.animation_progress = 0
            st.rerun()

        st.markdown("---")

        # Manual controls (compact grid)
        st.subheader("Manual moves")
        moves = ['F', 'R', 'U', 'B', 'L', 'D']
        primes = ["", "'", "2"]
        for mv in moves:
            cols = st.columns(3)
            for i, p in enumerate(primes):
                mvstr = mv + p
                if cols[i].button(mvstr, use_container_width=True):
                    st.session_state.pending_move = mvstr
                    st.session_state.animation_progress = 0.001
                    st.session_state.rotating_face = mv
                    st.session_state.clockwise = not mvstr.endswith("'")
                    st.session_state.auto_playing = False
                    # animation handled in-place (do not rerun here)

        st.markdown("---")

        # Solver controls (when available)
        if st.session_state.solution:
            st.subheader("Solver")
            st.write(f"Moves: {len(st.session_state.solution)}")
            st.text_area("Solution (compact)", " ".join(st.session_state.solution), height=80, key="solution_display")
            s1, s2, s3 = st.columns(3)
            with s1:
                if st.button("◀", use_container_width=True):
                    if st.session_state.current_step > 0:
                        st.session_state.current_step -= 1
                        st.session_state.auto_playing = False
                        temp_cube = RubiksCube()
                        for mvv in st.session_state.solution[:st.session_state.current_step]:
                            temp_cube.apply_move(mvv)
                        st.session_state.cube = temp_cube
                        st.rerun()
            with s2:
                if st.button("⏸", use_container_width=True):
                    st.session_state.auto_playing = False
                if st.button("▶︎•", use_container_width=True):
                    st.session_state.auto_playing = True
            with s3:
                if st.button("▶", use_container_width=True):
                    if st.session_state.current_step < len(st.session_state.solution):
                        mv2 = st.session_state.solution[st.session_state.current_step]
                        st.session_state.pending_move = mv2
                        st.session_state.animation_progress = 0.001
                        st.session_state.rotating_face = mv2[0]
                        st.session_state.clockwise = not mv2.endswith("'")
                        st.session_state.auto_playing = False

            st.markdown("---")

            # Professional informational expander explaining solver & performance tips
            with st.expander("About this app and how it solves the cube", expanded=False):
                st.markdown("**Overview**")
                st.markdown(
                    "- This app visualizes a 3×3 Rubik's Cube using Plotly Mesh3d and allows smooth layer rotations.\n"
                    "- The solver pipeline is defensive: it first tries an external solver, then an internal search-based fallback, and finally a simple move-history undo as last resort."
                )
                st.markdown("**Solver pipeline (brief)**")
                st.markdown(
                    "- External solver: when available the app converts the current state into the external library's Cube object and asks that solver for a sequence.\n"
                    "- Internal fallback: a lightweight IDA* implementation (configurable depth and time limit) is used if the external solver fails or is not present.\n"
                    "- Final fallback: if no full solution is found, the app can reverse recent moves from the recorded move history to recover the solved state."
                )
                st.markdown("**Internal solver details**")
                st.markdown(
                    "- Algorithm: IDA* (iterative deepening A*) with a simple misplaced-sticker heuristic.\n"
                    "- Limits: configured max depth and time-limit to avoid long blocking computations in the server.\n"
                    "- Purpose: reliable recovery for short scrambles and to provide a fallback when external solver isn't available."
                )
                st.markdown("**Performance tips for Streamlit Cloud**")
                st.write(
                    "- Reduce 'Smoothness (frames)' to 8–20 on small instances.\n"
                    "- Increase 'Animation speed' to shorten per-frame waits.\n"
                    "- Use the 2D net when running on very small VMs or mobile browsers.\n"
                    "- For production solving, consider integrating a dedicated two-phase solver (Kociemba) server-side."
                )
                st.markdown("**Contributing / Extending**")
                st.markdown(
                    "- Visualization: move animations client-side using Plotly frames or a small JS renderer to avoid server blocking.\n"
                    "- Solver: replace the internal fallback with a production solver for guaranteed fast solves.\n"
                    "- Tests: add unit tests for move application and solver correctness."
                )

    # Main content area - both visualizations
    colA, colB = st.columns([3, 2])
    
    with colA:
        st.markdown(
        "<h1 style='text-align: center;'>3D</h1>", 
            unsafe_allow_html=True
        )
        fig3 = create_3d_cube_visualization(
            st.session_state.cube.get_cube_state(),
            st.session_state.cube.color_map,
            st.session_state.animation_progress,
            st.session_state.rotating_face,
            st.session_state.clockwise
        )
        st.plotly_chart(fig3, use_container_width=True)

    with colB:
        st.markdown(
            "<h1 style='text-align: center;'>2D</h1>", 
            unsafe_allow_html=True
        )
        fig2 = create_2d_net_visualization(
            st.session_state.cube.get_cube_state(),
            st.session_state.cube.color_map
        )
        st.plotly_chart(fig2)

    # Handle animations
    # animation_progress is advanced each run; when it reaches 1.0 we apply pending_move
    if st.session_state.animation_progress and st.session_state.pending_move:
        # increment = (speed multiplier) * (1 / frames)
        increment = (st.session_state.animation_speed / 10.0) * (1.0 / max(1, st.session_state.animation_frames))
        st.session_state.animation_progress = min(1.0, st.session_state.animation_progress + increment)
        if st.session_state.animation_progress >= 1.0:
            # finalize move
            try:
                st.session_state.cube.apply_move(st.session_state.pending_move)
            except Exception:
                pass
            st.session_state.move_history.append(st.session_state.pending_move)
            # if this was an autoplayed solution move advance current_step
            if st.session_state.auto_playing and st.session_state.solution and st.session_state.current_step < len(st.session_state.solution):
                st.session_state.current_step += 1
            # if this was manual "Next" also advance
            elif not st.session_state.auto_playing and st.session_state.solution and st.session_state.current_step < len(st.session_state.solution) and st.session_state.pending_move == st.session_state.solution[st.session_state.current_step]:
                st.session_state.current_step += 1
            # reset animation state
            st.session_state.pending_move = None
            st.session_state.animation_progress = 0.0
            st.session_state.rotating_face = None
        # small sleep to avoid hammering reruns (keeps animation smooth)
        time.sleep(0.03)
        st.rerun()

    # Auto-play functionality
    if st.session_state.auto_playing and st.session_state.solution:
        if st.session_state.current_step < len(st.session_state.solution):
            # start pending move and let animation handler apply the move when complete
            mv = st.session_state.solution[st.session_state.current_step]
            if not st.session_state.pending_move:
                st.session_state.pending_move = mv
                st.session_state.animation_progress = 0.001
                st.session_state.rotating_face = mv[0]
                st.session_state.clockwise = not mv.endswith("'")
            time.sleep(0.03)
            st.rerun()
        else:
            st.session_state.auto_playing = False
            st.balloons()

if __name__ == "__main__":
    main()