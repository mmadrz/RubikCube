# RubikCube — Interactive Solver & Visualizer

A Streamlit-based interactive Rubik's Cube visualizer and solver with:
- 3D and 2D net visualizations (Plotly).
- Smooth animated layer rotations that keep cubies attached.
- Solver pipeline that uses a Rubik library (when available) and an internal IDA* fallback.
- Manual controls, scramble/reset, autoplay of solution, and basic move history.

This repository contains a single-file Streamlit app: `Rubik.py`.

Contents
- Rubik.py — main Streamlit app with visualization, animation pipeline and solver logic.
- Styles/ — CSS used by the Streamlit UI (logo, header, layout tweaks).
- logo.png — sidebar logo used in the app.

Quick start

1. Create and activate Python environment (recommended)
   - python3 -m venv .venv
   - source .venv/bin/activate  (Windows: .venv\Scripts\activate)

2. Install dependencies
   - pip install streamlit plotly numpy rubik-cube==<if used>    # replace rubik-cube with actual package name if used
   - (If a specific local rubik library is used, ensure it is installed or available in PYTHONPATH.)

3. Run the app
   - streamlit run Rubik.py
   - Open the URL shown in the terminal (usually http://localhost:8501).

User guide (UI overview)

- Sidebar
  - Logo and recent move history.
  - Animation Settings:
    - Number of Scramble Moves — how many random moves for scramble.
    - Animation Smoothness (frames) — higher = smoother but slower per move.
    - Animation Speed (multiplier) — higher = faster overall animation.
  - Cube Actions:
    - Scramble — random scramble (resets solution).
    - Solve — compute solution (uses rubik library if available; otherwise tries internal solver).
    - Reset Cube — returns to solved state.
  - Manual Controls:
    - Buttons for every face move: F, R, U, B, L, D and their variants ("", "'", "2").
    - Clicking launches a smooth animation and applies the move on completion.
  - Solver Controls (when solution exists):
    - Play / Pause / Next / Previous buttons and a compact solution display.

- Main content
  - Left: 3D Cube Visualization (Plotly). Uses cubie-grouped meshes so rotating layers move as real-world cube layers.
  - Right: 2D Cube Net for quick state reading.
  - Animations run in-place (placeholder) to avoid reloading the whole page per frame.

Solver details

- Primary: external rubik library integration (convert cube state to library format and call its solver).
- Fallback: Internal IDA* solver (simple heuristic — misplaced stickers) used when:
  - the library reports invalid/no-solution or raises exceptions,
  - or when the library is not installed.
- Final fallback: reverse recent move history (very simple undo) when no solver returns a plan.

Notes on the internal solver
- Small IDA* implementation tuned for short scrambles (configurable max_depth and time limits).
- Works for short / moderate scrambles; expected exponential runtime for deep scrambles.
- For production-level solving, integrate a two-phase algorithm (Kociemba) or use an optimized library.

Performance tips
- Reduce Animation Smoothness (frames) to speed up rendering on low-end machines.
- Increase Animation Speed if animations feel slow.
- The app batches geometry into cubie-based Mesh3d traces (27 meshes) to reduce browser overhead.
- For very constrained devices, disable animations (apply moves instantly) or use only the 2D net.

Developer notes
- Visualization: create_3d_cube_visualization builds cubie-grouped meshes and applies per-frame transforms to keep stickers attached.
- State: Streamlit `st.session_state` stores cube model, pending moves, animation flags and recent move history.
- To extend solver: add a stronger heuristic (pattern DB) or integrate an optimized solver (Kociemba).
- To offload animation to client: consider exporting Plotly frames/animations or a small JS renderer that accepts per-frame transforms.

Troubleshooting
- If colors or faces appear incorrect, check the face-to-color mapping in `RubiksCube.color_map` and the conversion functions `_color_to_char` / `_char_to_color`.
- If the app freezes during long solves, adjust `InternalSolver` time_limit and max_depth in `Rubik.py`.
- If the external rubik library import fails, either install it or allow the internal solver to run.

License & Attribution
- Add your preferred license here (e.g., MIT).
- Mention any third-party libraries and assets (Plotly, Streamlit, logo).

Contact / Contribution
- Open an issue or PR in this repository with bug reports or enhancement suggestions.
- Small, focused PRs (visual tweaks, solver improvements) are easiest to review.
