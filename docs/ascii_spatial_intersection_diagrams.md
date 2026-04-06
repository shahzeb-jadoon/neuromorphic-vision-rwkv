# Guidelines: Spatial Intersection Diagrams

When diagnosing autonomy and pathfinding behavior in complex junctions (like CARLA's Town06 or Town10HD), standard logs are often insufficient. Visualizing exact telemetry is required to differentiate between optical illusions (like curves rendering poles further away) and actual behavioral defects (like lateral lane-association bugs).

This project uses **High-Resolution Spatial ASCII Diagrams** to map out anomalous intersections precisely. These diagrams use fixed-width structural ASCII characters to ensure layout proportions correctly reflect physical HD map spacing (lane widths, road angles, crosswalks, and stops).

---

## 1. Core Principles

1.  **Fixed-Width Preservation**: Use only single-character ASCII/Unicode structural characters. Do not use emojis (like 🚗 or 🔴) for structures, as their rendered width varies by OS and font, tearing the structural grid alignment.
2.  **Double-Magnitude Scale**: The diagram must be large enough to visualize 5-10 meters of the roadway leading into *and* out of the intersection. One horizontal character ` ` space represents roughly 0.5 meters of lane width.
3.  **Holistic Context**: A diagram must show where the ego entered from, its exact trajectory *through* the junction, and the destination/downstream target (especially if the traffic light physically lies past the center of the junction).

---

## 2. Structural Palette

| Description | Character Sequence | Notes |
| :--- | :--- | :--- |
| **Median (Opposing Traffic)** | `════════╬════════` or `║` | Separates opposing lanes. Think of this as the double-yellow line. |
| **Lane Divider (Same Direction)** | `┈┈┈┈┈┈┈┈` or `╵` / `╷` | Separates lanes flowing in the same direction. Think of this as the white dashed line. |
| **Stop Line / Crosswalk** | `════════╪════════` | Usually placed symmetrically across the entry/exit lanes of the junction. |
| **Outer Road Boundaries** | `─────────────────────┘` | The physical curb or grass edge of the road map. |
| **Ego Start/Stop** | `[E:START]`, `[EGO ]` | Shows the timeline of the vehicle's position. |
| **Target Actor** | `[TL:RED] ▼`, `[VEH]` | Fixed environment variables causing the behavior. |
| **Measured Path (Actual)** | `. . . . . ` | The trajectory the car *actually* took. |
| **Expected Path** | `~ ~ ~ ~ ~ ` | The trajectory the car *should have* taken. |

---

## 3. How to Draw a Multi-Lane Junction

To draft a robust diagnostic diagram, follow this scaffolding order:

### Step 1: Lay the Road Boundaries and Lane Geometry
Start by drawing the four directions (or three for a T-Junction). Ensure you give each direction a double-line median and dashed dividers for multiple lanes.
```text
                               │        ╵        ║        ╵        │
                               │   ⬇    ╵   ⬇    ║   ⬆    ╵   ⬆    │
                               │        ╵        ║        ╵        │
                               │════════╪════════╬════════╪════════│
   ────────────────────────────┘                 ╚                 └────────────────────────────
        Westbound ⬅                                                     Westbound ⬅
   ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈                                     ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
        Westbound ⬅                                                     Westbound ⬅
   ════════════════════════════                                     ════════════════════════════
        Eastbound ➡                                                     Eastbound ➡
   ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈                                     ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
        Eastbound ➡                                                     Eastbound ➡
   ────────────────────────────┐                 ╔                 ┌────────────────────────────
                               │        ╷        ║        ╷        │
                               │   ⬇    ╷   ⬇    ║   ⬆    ╷   ⬆    │
                               │        ╷        ║        ╷        │
```

### Step 2: Superimpose Traffic Controls
Locate the Traffic Light poles and Expected Stop lines relative to the Ego's trajectory. Place them inside the grid. Be precise—if a matched stop line was for the far-left lane, draw it only in the far-left lane block.

### Step 3: Draw the Actor Path
Plot the beginning of the incident (`[E:START]`), the dots marking its travel `. . . .`, and the terminating point (`[EGO ]`).
*   **Crucial:** Make sure the dots reflect the literal (x,y) curvature from the telemetry, tracking lane-drift if applicable.

### Step 4: Add Context Indicators
If the ego's behavior is incorrect, plot the *true expectation* using tildes `~ ~ ~ ~` to immediately highlight the delta between the logic branch taken and the correct logic branch.

---

## 4. Example Incident (Lateral Lane Drift)

```text
                               │        ╵        ║        ╵        │
                               │   ⬇    ╵   ⬇    ║   ⬆    ╵   ⬆    │
                               │        ╵        ║        ╵        │
                               │════════╪════════╬════════╪════════│
   ────────────────────────────┘                 ╚                 └────────────────────────────
        Westbound ⬅                                                     Westbound ⬅
   ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
        Westbound ⬅                   ~ ~ ~ ~ [TARGET_LANE]             Westbound ⬅
   ════════════════════════════      ~                              ════════════════════════════
        Eastbound ➡ (Left Lane)     ~    [EGO ]                     Eastbound ➡
   ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈    ~     . . . .                    ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
        Eastbound ➡ (Right Lane)  ~              . . . [E:START]    Eastbound ➡
   ────────────────────────────┐                 ╔                 ┌────────────────────────────
                               │        ╷        ║        ╷        │
```
