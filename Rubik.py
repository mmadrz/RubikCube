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
            'R': np.full((3, 3), 're09d', dtype='<U6'),      # Right  
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


def create_3d_cube_visualization(cube_state, color_map, animation_phase=0, rotating_face=None, clockwise=True):
    fig = go.Figure()
    
    # Define face positions and orientations
    face_positions = {
        'F': {'center': [0, 0, 1.5], 'normal': [0, 0, 1], 'up': [0, 1, 0]},
        'R': {'center': [1.5, 0, 0], 'normal': [1, 0, 0], 'up': [0, 1, 0]},
        'B': {'center': [0, 0, -1.5], 'normal': [0, 0, -1], 'up': [0, 1, 0]},
        'L': {'center': [-1.5, 0, 0], 'normal': [-1, 0, 0], 'up': [0, 1, 0]},
        'U': {'center': [0, 1.5, 0], 'normal': [0, 1, 0], 'up': [0, 0, 1]},
        'D': {'center': [0, -1.5, 0], 'normal': [0, -1, 0], 'up': [0, 0, -1]}
    }
    
    for face_char, face_data in cube_state.items():
        face_info = face_positions[face_char]
        for i in range(3):
            for j in range(3):
                color = color_map.get(face_data[i, j], "#000000")
                
                # Calculate sticker position
                x = j - 1
                y = 1 - i
                z = 0
                
                # Position the sticker on the face
                if face_char in ['F', 'B', 'L', 'R']:
                    if face_char == 'F':
                        pos = [x, y, 1.5]
                    elif face_char == 'B':
                        pos = [-x, y, -1.5]
                    elif face_char == 'R':
                        pos = [1.5, y, -x]
                    elif face_char == 'L':
                        pos = [-1.5, y, x]
                else:  # U or D
                    if face_char == 'U':
                        pos = [x, 1.5, -y]
                    else:  # D
                        pos = [x, -1.5, y]
                
                # Apply animation rotation if this sticker is part of the rotating face
                if rotating_face is not None and animation_phase > 0:
                    if _is_sticker_on_face(face_char, i, j, rotating_face):
                        angle = animation_phase * 90 * (-1 if clockwise else 1)
                        pos = _rotate_sticker(pos[0], pos[1], pos[2], rotating_face, angle)
                
                add_sticker(fig, pos, face_info['normal'], color)
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.8, 1.8]),
            yaxis=dict(visible=False, range=[-1.8, 1.8]),
            zaxis=dict(visible=False, range=[-1.8, 1.8]),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
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
                return []
            
            # Solve using the library
            solver = Solver(rubik_cube)
            solver.solve()
            
            # Check if solution was found
            if not solver.moves:
                st.warning("No solution found - cube might already be solved")
                return []
            
            # Optimize the moves
            optimized_moves = optimize_moves(solver.moves)
            
            return optimized_moves
            
        except Exception as e:
            # Fallback to a simple scramble reversal
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
        st.session_state.animation_phase = 0
        st.session_state.rotating_face = None
        st.session_state.clockwise = True
        st.session_state.animation_speed = 0.5

    # Sidebar with all controls and information
    with st.sidebar:

            # Prepare ditimo logo as base64
        with open("logo.png", "rb") as img_file:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode()

            # Use the styled HTML structure
            st.markdown(
                f"""
                <div class="neon-border-container">
                    <div class="neon-border">
                        <img src="data:image/png;base64,{img_base64}" />
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        # Move History
        if st.session_state.move_history:

            st.markdown(
                "<h1 style='text-align: center;'>📝 Recent Moves</h1>", 
                    unsafe_allow_html=True
                )
            st.write(" → ".join(st.session_state.move_history[-100:]))
                
        # Animation Settings
        st.markdown(
        "<h1 style='text-align: center;'>⚙️ Animation Settings</h1>", 
            unsafe_allow_html=True
        )
        st.session_state.scramble_moves = st.slider(
            "Number of Scramble Moves",
            min_value=5,
            max_value=100,
            value=20,
            step=1,
            help="Select the number of random moves for scrambling the cube"
        )
        st.session_state.animation_speed = st.slider(
            "Animation Speed", 
            min_value=1, 
            max_value=50, 
            value=5, 
            step=1,
            help="Adjust the speed of cube rotations"
        )
        
        st.markdown("---")
        
        # Cube Actions Section
        st.markdown(
        "<h1 style='text-align: center;'>🎮 Cube Actions</h1>", 
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Scramble", use_container_width=True):
                moves = st.session_state.cube.scramble(st.session_state.scramble_moves)
                st.session_state.move_history = moves.copy()
                st.session_state.solution = []
                st.session_state.current_step = 0
                st.session_state.auto_playing = False
                st.session_state.animation_phase = 0
                st.rerun()
        with col2:
            if st.button("✅ Solve", use_container_width=True):
                with st.spinner("Solving cube..."):
                    solver = RubiksSolver()
                    solution = solver.solve(st.session_state.cube)
                    if solution:
                        st.session_state.solution = solution
                        st.session_state.current_step = 0
                        st.session_state.auto_playing = True
                    else:
                        st.session_state.solution = []
                        st.session_state.auto_playing = False
        
        if st.button("🔄 Reset Cube", use_container_width=True):
            st.session_state.cube = RubiksCube()
            st.session_state.solution = []
            st.session_state.current_step = 0
            st.session_state.auto_playing = False
            st.session_state.move_history = []
            st.session_state.animation_phase = 0
            st.rerun()
        
        st.markdown("---")
        
        # Manual Controls Section
        st.markdown(
        "<h1 style='text-align: center;'>🎯 Manual Controls</h1>", 
            unsafe_allow_html=True
        )
        moves = ['F','R','U','B','L','D']
        primes = ["", "'", "2"]
        
        for mv in moves:
            cols = st.columns(3)
            for i, p in enumerate(primes):
                mvstr = mv + p
                if cols[i].button(mvstr, use_container_width=True):
                    # Start animation
                    st.session_state.animation_phase = 0.1
                    st.session_state.rotating_face = mv
                    st.session_state.clockwise = not mvstr.endswith("'")
                    
                    # Apply the move to update both visualizations
                    st.session_state.cube.apply_move(mvstr)
                    st.session_state.move_history.append(mvstr)
                    st.session_state.auto_playing = False
                    
                    st.rerun()
        
        st.markdown("---")
        
        # Solver Controls Section
        if st.session_state.solution:
            st.header("⚙️ Solver Controls")
            
            st.write(f"**Total moves:** {len(st.session_state.solution)}")
            
            # Display moves in a more compact format
            moves_display = " ".join(st.session_state.solution)
            st.text_area("Solution Sequence", moves_display, height=100, key="solution_display")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("⏮ Previous", use_container_width=True):
                    if st.session_state.current_step > 0:
                        st.session_state.current_step -= 1
                        st.session_state.auto_playing = False
                        # Rebuild cube state for both visualizations
                        temp_cube = RubiksCube()
                        for mvv in st.session_state.solution[:st.session_state.current_step]:
                            temp_cube.apply_move(mvv)
                        st.session_state.cube = temp_cube
                        st.rerun()
            with c2:
                if st.button("⏸ Pause", use_container_width=True):
                    st.session_state.auto_playing = False
                    st.rerun()
                if st.button("▶ Play", use_container_width=True):
                    st.session_state.auto_playing = True
                    st.rerun()
            with c3:
                if st.button("⏭ Next", use_container_width=True):
                    if st.session_state.current_step < len(st.session_state.solution):
                        mv2 = st.session_state.solution[st.session_state.current_step]
                        st.session_state.cube.apply_move(mv2)
                        st.session_state.current_step += 1
                        st.session_state.auto_playing = False
                        st.rerun()
            
            # Current step progress
            if st.session_state.solution:
                st.progress(st.session_state.current_step / len(st.session_state.solution))
                st.write(f"Step {st.session_state.current_step + 1} of {len(st.session_state.solution)}")
        
        st.markdown("---")
        
        # Information Section
        st.markdown("<h1 style='text-align: center;'>ℹ️ Solver Information</h1>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<h3 style='text-align: center;'>🧊 Rubik's Cube Solver with Fallback System</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'><strong>Academic Project | Data Science Course</strong></p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'><em>University of Chinese Academy of Sciences</em></p>", unsafe_allow_html=True)

        st.markdown("---")

        with st.container():
            st.markdown("<p style='text-align: center;'><strong>Move Notation:</strong></p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Clockwise: F, R, U, B, L, D</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Counter-clockwise: F', R', U', B', L', D'</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>180° turns: F2, R2, U2, B2, L2, D2</p>", unsafe_allow_html=True)

    # Main content area - both visualizations
    colA, colB = st.columns([3, 2])
    
    with colA:
        st.markdown(
        "<h1 style='text-align: center;'>3D Cube Visualization</h1>", 
            unsafe_allow_html=True
        )
        fig3 = create_3d_cube_visualization(
            st.session_state.cube.get_cube_state(),
            st.session_state.cube.color_map,
            st.session_state.animation_phase,
            st.session_state.rotating_face,
            st.session_state.clockwise
        )
        st.plotly_chart(fig3, use_container_width=True)

    with colB:
        st.markdown(
            "<h1 style='text-align: center;'>2D Cube Net</h1>", 
            unsafe_allow_html=True
        )
        fig2 = create_2d_net_visualization(
            st.session_state.cube.get_cube_state(),
            st.session_state.cube.color_map
        )
        st.plotly_chart(fig2)

    # Handle animations
    if st.session_state.animation_phase > 0:
        st.session_state.animation_phase += 0.05 * st.session_state.animation_speed
        if st.session_state.animation_phase >= 1:
            st.session_state.animation_phase = 0
            st.session_state.rotating_face = None
        time.sleep(.5)
        st.rerun()

    # Auto-play functionality
    if st.session_state.auto_playing and st.session_state.solution:
        if st.session_state.current_step < len(st.session_state.solution):
            mv = st.session_state.solution[st.session_state.current_step]
            
            # Start animation for this move
            st.session_state.animation_phase = 0.1
            st.session_state.rotating_face = mv[0]
            st.session_state.clockwise = not mv.endswith("'")
            
            # Apply the move to update both visualizations
            st.session_state.cube.apply_move(mv)
            st.session_state.move_history.append(mv)
            st.session_state.current_step += 1
            
            time.sleep(0.8 / st.session_state.animation_speed)
            st.rerun()
        else:
            st.session_state.auto_playing = False
            st.balloons()

if __name__ == "__main__":
    main()