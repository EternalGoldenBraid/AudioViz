# Two cameras as equivalent evidence on one canvas

## World

Observers (cameras) sit at unknown poses $T_k$ in a world frame $W$ and observe features $X_j$ with unknown intrinsics $\pi_k$:

$$x_{kj} = \pi_k\!\left(T_k^{-1} X_j\right)$$

![Observers in a world frame observing 3D features](./model.svg)

The two cameras are the same kind of observer. They differ only in $T_k$ and $\pi_k$.

## Model

One shared hidden canvas $h$. Each camera $k$ has its own learned readout:

$$\hat{x}_k = g_k(h), \qquad e_k = x_k - \hat{x}_k$$

The canvas is corrected by the errors of every enabled camera at once. Cameras differ only through $g_k$ (pose and intrinsics are absorbed into it). No geometry is assumed.

## Phase 1 (multi-device agent)

- A second readout branch for camera 2 over the same canvas.
- Cover camera 1 and camera 2 still corrects the canvas.
- No coupling between branches; no homography.

## Phase 2: correspondence

Hidden units excited by pixels of both cameras are the correspondence.

1. For each hidden unit $i$, compute its receptive-field centroid $c_i^{(k)}$ in camera $k$ from $g_k$.
2. Keep units with strong, compact fields in both cameras.
3. Fit $H$ so that $c_i^{(2)} \simeq H\,c_i^{(1)}$ (DLT with RANSAC).
4. The residual is the diagnostic: low means a homography suffices.

![Hidden units as correspondences between the two cameras](./correspondence.svg)

## Open questions

- A homography is exact only for planar or distant scenes. What is the next transform (epipolar / fundamental matrix, depth-dependent warp, learned flow)?
- EO and IR are not a matched pair. Do shared units emerge at all without cross-modal supervision?
- Should the checkerboard calibration be an independent baseline?
- Where does the diagnostic show up in the TUI: tile, overlay or log?
