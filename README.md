# 🧊 RubikCube — Interactive 3×3 Visualizer & Solver

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-brightgreen?logo=streamlit)

A professional web application for visualizing and solving 3×3 Rubik's Cubes with a robust multi-tier solver pipeline.

**Live Demo:** [https://rubikcube.streamlit.app/](https://rubikcube.streamlit.app/)

---

## 🎯 Overview

| Aspect | Details |
|--------|---------|
| **Language** | Python 3.8+ |
| **UI Framework** | Streamlit |
| **3D Rendering** | Plotly (Mesh3d) |
| **State Model** | NumPy |
| **Main File** | `Rubik.py` |
| **Solver** | External (primary) + IDA* (fallback) + Move-history (final) |

> [!NOTE]
> This project is optimized for **local deployment**. The live Streamlit Cloud demo has performance limitations due to resource constraints. For the best experience, clone the repo and run locally.

---

## ✨ Features

- ✅ **Accurate 3×3 Cube Model** — Fully correct state representation and move application
- ✅ **Interactive 3D Visualization** — Smooth rotations with Plotly Mesh3d (cubie-grouped geometry)
- ✅ **2D Net View** — Compact planar representation for quick reference
- ✅ **Multiple Input Methods** — Manual controls, scramble, auto-play solutions
- ✅ **Robust Solver Pipeline** — External solver → IDA* fallback → Move-history recovery
- ✅ **Step-by-Step Playback** — Pause, play, next, previous controls
- ✅ **Client-Side Animations** — Pre-cached GIFs for smooth, non-blocking animations
- ✅ **Responsive Design** — Works on desktop and tablet browsers

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mmadrz/RubikCube.git
   cd RubikCube
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**

   ```bash
   streamlit run Rubik.py
   ```

5. **Open in browser:**
   Streamlit will print a URL (typically `http://localhost:8501`). Open it in your browser.

> [!TIP]
> On first run, Streamlit may prompt you to enter an email for analytics. You can skip this safely.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         Streamlit UI (Frontend)             │
│  ┌──────────────────────────────────────┐   │
│  │ Manual Controls | Solver Controls    │   │
│  │ Scramble | Reset | Play/Pause/Next   │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│       Visualization Layer (Plotly)          │
│  ┌──────────────────────────────────────┐   │
│  │ 3D Mesh3d (grouped per cubie)        │   │
│  │ 2D Net (fixed, non-interactive)      │   │
│  │ Animation Pipeline (GIF playback)    │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      Cube Logic & Solver Layer              │
│  ┌──────────────────────────────────────┐   │
│  │ RubiksCube (state model)             │   │
│  │ ├─ External Solver (rubik-cube lib)  │   │
│  │ ├─ IDA* Fallback (internal)          │   │
│  │ └─ Move-History Recovery             │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

---

## 🧩 How the Solver Works

The solver uses a **three-tier fallback pipeline** for robustness:

### 1️⃣ **External Solver (Primary)**

- Converts the internal cube state to the `rubik-cube` library format.
- Calls the external solver for a fast, optimized solution.
- **Pros:** Very fast, production-quality.
- **Cons:** Depends on external library availability.

### 2️⃣ **Internal IDA* Fallback (Secondary)**

- **Algorithm:** Iterative Deepening A* with a misplaced-sticker heuristic.
- **Heuristic:** Counts stickers not matching their face center, scaled down by 8 to keep it admissible.
- **Limits:**
  - Maximum depth: 20 moves
  - Time limit: 6–8 seconds (configurable)
- **Pros:** Deterministic, no external dependencies, safe resource usage.
- **Cons:** Slower than production solvers for complex scrambles.

### 3️⃣ **Move-History Recovery (Final Fallback)**

- If both solvers fail, reverses the recent move history to return to the solved state.
- **Pros:** Always succeeds (worst-case scenario).
- **Cons:** Not an optimal solution; purely functional.

### Why This Pipeline?

```
✓ Robustness:  Falls back gracefully if any tier fails
✓ Speed:       Uses fastest available solver first
✓ Safety:      Limits prevent server overload
✓ UX:          User always gets *some* result
```

---

## 📊 Visualization Details

### 3D Rendering

- **Geometry:** 54 sticker quads grouped by cubie (fewer Plotly traces = better performance).
- **Rotation:** Smooth interpolation during moves (0–90° per frame).
- **Lighting:** Phong lighting for depth perception.
- **Camera:** Fixed isometric view for clarity.

### 2D Net

- **Layout:** Fixed cross-shaped net (U top, L-F-R-B middle row, D bottom).
- **Interaction:** Disabled (fixed view to prevent confusion).
- **Purpose:** Quick reference; useful on resource-constrained devices.

---

## ⚙️ Configuration & Tuning

### Sidebar Controls

| Control | Range | Effect |
|---------|-------|--------|
| **Scramble Length** | 5–100 moves | Number of random moves for scrambling |
| **Smoothness (frames)** | 4–60 frames | Animation interpolation steps (higher = smoother but slower) |
| **Animation Speed** | 1–100× | Multiplier for animation frame delay |

### Performance Tips for Streamlit Cloud

> [!WARNING]
> The live demo may experience lag due to shared server resources. For best performance, **run locally**.

- **Reduce Smoothness:** Use 4–12 frames for faster animation rendering.
- **Increase Speed:** 50–100× for snappier UI feedback.
- **Use 2D Net Only:** Disable 3D rendering if you need minimal server load.
- **Disable Auto-Play:** Manual step-through is lighter on the server.

---

## 📁 Code Structure

```
Rubik.py
├── RubiksCube
│   ├── is_solved()
│   ├── rotate_face() / _rotate_adjacent_edges()
│   ├── apply_move()
│   ├── scramble()
│   ├── to_rubik_lib_cube() / from_rubik_lib_cube()
│   └── clone()
│
├── InternalSolver
│   ├── _heuristic()
│   ├── _inverse()
│   └── solve() [IDA*]
│
├── Visualization Functions
│   ├── create_3d_cube_visualization()
│   ├── create_2d_net_visualization()
│   ├── _rotate_sticker()
│   └── add_sticker()
│
├── RubiksSolver (wrapper)
│   ├── solve() [three-tier pipeline]
│   ├── _is_valid_cube()
│   └── _fallback_solution()
│
└── main() [Streamlit app entry]
    ├── Session state initialization
    ├── Sidebar UI (controls, settings)
    ├── Main visualizations
    └── Animation & autoplay loops
```

---

## 🔧 Development & Contributing

### Running Tests

```bash
# (Add unit tests in tests/ directory)
pytest tests/
```

### Future Improvements

- [ ] **Plotly Frames Animation** — Replace GIFs with client-side Plotly frame playback (vector/WebGL).
- [ ] **Kociemba Solver** — Integrate a production-grade two-phase solver for optimal solutions.
- [ ] **Unit Tests** — Add comprehensive tests for `apply_move()` and solver correctness.
- [ ] **3D Controls** — Optional mouse-drag cube rotation for exploration.
- [ ] **Solution Statistics** — Show move count, execution time, optimality.

### Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push and open a pull request.

> [!NOTE]
> Please include screenshots for visual changes and test coverage for logic updates.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"ModuleNotFoundError: No module named 'rubik'"** | Run `pip install -r requirements.txt` to install all dependencies. |
| **Animations are choppy or lag** | Reduce "Smoothness (frames)" slider (try 4–12). Increase "Animation Speed" (try 50+). |
| **3D visualization doesn't render** | Check that Plotly is installed. Try restarting the app. |
| **"Solver timed out" message** | The internal solver exceeded the time limit. Try a simpler scramble (fewer moves). |
| **GIF generation fails on Streamlit Cloud** | This is a kaleido/memory issue. Reduce animation frames or run locally instead. |

> [!TIP]
> For best results, **always run locally** rather than relying on the cloud demo.

---

## 📦 Dependencies

```
streamlit>=1.0
numpy>=1.20
plotly>=5.0
kaleido>=0.2        # For GIF generation
imageio>=2.0        # For GIF assembly
Pillow>=8.0         # Image handling
rubik-cube>=1.0     # External solver library
```

See `requirements.txt` for exact pinned versions.

---

## 📄 License

This project is licensed under the **MIT License**.

```
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
```

---

**Built with ❤️ by [Mohammadreza Fathi](https://github.com/yourusername)**
