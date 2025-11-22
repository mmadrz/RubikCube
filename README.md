# RubikCube — Interactive Visualizer & Solver

Professional, interactive Rubik's Cube web app built with Streamlit and Plotly.  
Features a crisp 3D visualization with physically-correct cubie rotations, a 2D net view, manual controls, scramble/solve, and a solver pipeline that uses an external solver with an internal IDA* fallback.

Badges
- Streamlit · Plotly · NumPy

Quick summary
- Language: Python
- UI: Streamlit
- 3D renderer: Plotly Mesh3d (cubie-grouped meshes)
- Solver: External rubik library (primary) + Internal IDA* fallback (secondary)
- File: main app — `Rubik.py`

Why this project
- Educational demo of cube state modelling, visualization and simple search-based solving.
- Clean UI for teaching cube notation and step-by-step solution playback.
- Modular design so visualization and solver can be improved independently.

Interactive demo (what to try)
1. Click "Scramble" to randomize the cube (choose scramble length in sidebar).
2. Use manual controls (F, R, U, B, L, D with '', ', 2) to apply moves and observe smooth animations.
3. Click "Solve" — the app uses an external solver if present; otherwise it will try the internal IDA* fallback.
4. Use Play / Pause / Next / Previous to step through the computed solution with animations.

Installation

1. Create a virtual environment (recommended)
   - python -m venv .venv
   - Windows: .venv\Scripts\activate
   - macOS / Linux: source .venv/bin/activate

2. Install requirements
   - pip install -r requirements.txt
   - If you have a different rubik package, install that and ensure names in requirements match.

3. Run the app
   - streamlit run Rubik.py
   - Open the local URL shown by Streamlit (usually http://localhost:8501).

User interface (concise)

- Sidebar
  - Logo and recent moves
  - Animation Settings:
    - Scramble length, Animation frames (smoothness), Animation speed (multiplier)
  - Cube Actions: Scramble, Solve, Reset
  - Manual Controls: Per-face move buttons (including primes and doubles)
  - Solver Controls: Play / Pause / Next / Previous and solution text area

- Main area
  - Left: 3D cube (interactive camera disabled for consistent view)
  - Right: 2D net (fixed, non-interactive)
  - Move history and progress shown in the sidebar

Solver internals (short)
- Primary: Attempts to convert the app state to the external `rubik` library cube and run its solver.
- Secondary: If the external solver fails, the app runs a compact IDA* search (InternalSolver) using a simple misplaced-sticker heuristic. This is intentionally conservative (configurable depth/time limit) and works for short scrambles.
- Final fallback: If no solution is found, the app can reverse recent moves (move-history undo) as a last resort.

Performance tuning (practical)
- Reduce "Animation Smoothness (frames)" to lower client work.
- Increase "Animation Speed" to shorten per-move delays.
- Use the 2D net only for very slow machines (less GPU/JS overhead).
- For best client performance, use modern browsers (Chrome/Edge) and avoid very large frame counts.

Developer guide (for contributors)
- Main logic: `Rubik.py`
  - RubikCube: cube state and move application
  - Visualization: `create_3d_cube_visualization` and `create_2d_net_visualization`
  - Solver wrapper: `RubiksSolver` uses external and internal solver
  - Animation: placeholder-based in-place updates (avoids repeated Streamlit reruns)
- Tips to extend:
  - Integrate a Kociemba / two-phase solver for production-grade solving.
  - Move animation to client-side (Plotly frames or minimal JS) to eliminate server-side blocking.
  - Add keyboard shortcuts or hotkeys for manual moves (requires JS integration).

Troubleshooting
- If faces look incorrect: verify mappings in `_color_to_char` / `_char_to_color` and `color_map`.
- If external solver import fails: either install the required `rubik` package or rely on the internal solver.
- If app freezes during long solves: reduce `InternalSolver` time_limit or increase compute resources.

Security & licensing
- No sensitive network calls are made by default.
- Add your preferred license (e.g., MIT) and include attribution for third-party assets.

Contributing
- Fork, branch, and open PRs. Small, well-scoped changes are easiest to review.
- Please include screenshots for UI/visual changes and tests for solver updates where applicable.

Contact
- Add contact info or GitHub repo issues link here.

Acknowledgements
- Built using Streamlit, Plotly and NumPy.

Enjoy exploring and improving the cube visualizer!
