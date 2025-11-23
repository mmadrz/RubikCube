# RubikCube — Interactive Visualizer & Solver

Professional web app that visualizes a 3×3 Rubik's Cube and computes solutions.  
Built with Streamlit (UI), Plotly (3D/2D rendering) and NumPy (state model). Live demo: https://rubikcube.streamlit.app/

Summary
- Language: Python
- UI: Streamlit
- 3D renderer: Plotly Mesh3d (cubie-grouped meshes)
- Solver: External rubik solver (primary) + Internal IDA* fallback (secondary)
- Main file: `Rubik.py`

Quick start
1. Create and activate a virtualenv:
   - python -m venv .venv
   - Windows: .venv\Scripts\activate
   - macOS / Linux: source .venv/bin/activate
2. Install dependencies:
   - pip install -r requirements.txt
3. Run:
   - streamlit run Rubik.py
4. Open the app in the browser (Streamlit prints the URL, usually http://localhost:8501).

Live demo
- Deployed on Streamlit Cloud: https://rubikcube.streamlit.app/

What this project provides
- Accurate cube state model (RubiksCube) and move application.
- 3D visualization with grouped cubie meshes and physically-correct rotations.
- 2D net view for a compact representation.
- Manual controls, scramble/reset, solver pipeline and step-by-step solution playback.
- Performance-oriented animation pipeline (client-playable assets and in-place updates).

How solving works (method)
1. External solver (preferred)
   - The app attempts to convert the internal cube state to the external library's cube object and invoke its solver. This is fast and reliable when available.
2. Internal fallback: IDA* search
   - If the external solver is missing or fails, the app runs a compact IDA* (Iterative Deepening A*) search as a fallback.
   - Heuristic: count of misplaced stickers compared to each face center, scaled down to keep it admissible-ish for shallow searches.
   - Limits: configurable maximum depth and time limit to avoid long-running server work.
   - Purpose: recover solutions for short scrambles or to provide a deterministic fallback in environments lacking the external solver.
3. Final fallback: move-history undo
   - If neither solver returns a solution, the app can reverse recently applied moves (from the recorded move history) to return to a solved state.

Why this pipeline
- Robustness: external solvers are strongest; internal search is a safety net.
- Control: time/depth limits prevent server blocking.
- User experience: clear feedback and a workable fallback even when environment constraints exist.

Visualization & performance (important for Streamlit Cloud)
- Avoid repeated Streamlit reruns during animations:
  - Visual updates use placeholders (st.empty) to update charts in-place instead of forcing reruns.
- Client-playable animations (recommended):
  - Per-move animations are pre-rendered and cached (small sequence of PNG frames assembled into a GIF) so the browser plays the animation locally and the server does not serve each frame.
  - Benefits: reduced server CPU use, no frequent network updates, smoother perceived animation.
- Caching:
  - Heavy geometry and generated GIFs are cached keyed by (serialized cube state + move) to avoid recomputation.
- Tunables for low-resource deployments:
  - Reduce "Smoothness (frames)" slider.
  - Increase "Animation speed" to shorten per-frame delays.
  - Use the 2D net-only mode for minimal rendering cost.
- Future improvement path:
  - Replace GIFs with Plotly frames and client-side animation (best UX, vector/WebGL), or use a small JS renderer for fully client-driven animations.

Code structure (concise)
- Rubik.py
  - RubiksCube: cube state, move application, serialization helpers
  - create_3d_cube_visualization: builds the Plotly Mesh3d representation (grouped per cubie)
  - create_2d_net_visualization: static planar net rendering (fast)
  - InternalSolver: IDA* fallback implementation
  - RubiksSolver: wrapper coordinating external solver, internal fallback and move-history fallback
  - Animation pipeline: placeholder-based updates and cached per-move animations

Design decisions and trade-offs
- Mesh grouping: stickers grouped per cubie (fewer traces) to reduce Plotly overhead.
- Caching vs memory: cache sizes tuned to balance repeated reuse and memory consumption on small VMs.
- Conservative internal solver: avoids long blocking operations on the shared server; production-grade solving should integrate a dedicated, optimized solver (e.g., Kociemba/Kociemba2-phase).

Deployment
- Deploy to Streamlit Cloud or any server that supports Streamlit.
- Recommended small VM settings:
  - Keep image export size reasonable (e.g., <= 480px) for GIF generation on the server.
  - Use fewer frames to reduce generation time and GIF size.
- Live deployed URL: https://rubikcube.streamlit.app/

Developer notes
- To add client-frame animations (Plotly frames) you will:
  - Build a single Plotly figure with a frames array and a small JS control to autoplay frames in the browser.
  - This eliminates server-side image export and produces smooth WebGL animations.
- To optimize geometry further:
  - Merge sticker quads into as few Mesh3d traces as possible.
  - Avoid per-sticker traces that cause heavy rendering overhead.
- Testing:
  - Add unit tests for move application (apply_move) and round-trip conversions to/from external cube formats.

License — MIT
Copyright (c) 2025 Mohammadreza Fathi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Contact & issues
- Use the repository issues to report bugs or request features.
