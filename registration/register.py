#!/usr/bin/env python3
"""
Lesion-aware non-rigid image registration pipeline.

Aligns a SOURCE image to a REFERENCE image so that a specific structure
(wound, mole, lesion) retains its relative size while the surrounding skin
is warped into alignment.

Architecture
------------
1. Feature extraction & matching  — SuperPoint + LightGlue
2. Mask-based keypoint filtering  — keypoints inside the (dilated) lesion mask
                                    are discarded so the warp never "explains
                                    away" the lesion's size change
3. Non-rigid warping              — Thin Plate Spline (cv2.ThinPlateSpline)
4. Visualisation                  — side-by-side + blended overlay + diff map

Usage
-----
    python register.py reference.jpg source.jpg \
        --mask source_mask.png \
        --dilation 15 \
        --out-warped warped.png \
        --out-vis visualisation.png

Requirements
------------
    pip install lightglue torch torchvision opencv-python numpy
    (LightGlue: https://github.com/cvg/LightGlue)
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

# LightGlue — official cvg implementation
# pip install git+https://github.com/cvg/LightGlue.git
from lightglue import ALIKED, DISK, LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# ---------------------------------------------------------------------------
# Global defaults  (all overridable via CLI or direct function arguments)
# ---------------------------------------------------------------------------
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_DILATION   = 15     # px — safety buffer around the lesion
MAX_KEYPOINTS   = 4096   # keypoint cap (increase for more matches)
MIN_MATCHES     = 6      # minimum matches required to compute a TPS warp
OVERLAY_ALPHA   = 0.5    # blend weight for the overlay visualisation panel

# Supported feature extractor names
FEATURES_CHOICES = ["superpoint", "disk", "aliked-n16", "aliked-n32"]
DEFAULT_FEATURES = "disk"  # DISK is generally most robust on skin/texture imagery


# ===========================================================================
# 1.  Feature extraction and matching
# ===========================================================================

def _build_extractor(
    features: str,
    max_keypoints: int,
    device: torch.device,
):
    """Instantiate the requested feature extractor.

    Supported values for *features*:
      - ``superpoint``  — SuperPoint (fast, good for structured scenes)
      - ``disk``        — DISK (learned, robust on texture/skin, recommended)
      - ``aliked-n16``  — ALIKED small (fast learned detector)
      - ``aliked-n32``  — ALIKED large (best accuracy, slower)
    """
    features = features.lower()
    if features == "superpoint":
        return SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    if features == "disk":
        return DISK(max_num_keypoints=max_keypoints).eval().to(device)
    if features in ("aliked-n16", "aliked-n32", "aliked"):
        model_name = "aliked-n32" if "32" in features else "aliked-n16"
        return ALIKED(model_name=model_name, max_num_keypoints=max_keypoints).eval().to(device)
    raise ValueError(
        f"Unknown feature extractor '{features}'. "
        f"Choose from: {FEATURES_CHOICES}"
    )


def extract_and_match(
    ref_path: str,
    src_path: str,
    features: str = DEFAULT_FEATURES,
    device: torch.device = DEVICE,
    max_keypoints: int = MAX_KEYPOINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features and match them with LightGlue.

    Parameters
    ----------
    ref_path      : path to the reference (fixed) image
    src_path      : path to the source (moving) image
    features      : feature extractor to use (superpoint/disk/aliked-n16/aliked-n32)
    device        : torch device
    max_keypoints : maximum keypoints per image

    Returns
    -------
    ref_pts : (N, 2) float32 — matched keypoints in the reference image
    src_pts : (N, 2) float32 — corresponding keypoints in the source image
    ref_bgr : H×W×3 uint8   — reference image in BGR for OpenCV
    src_bgr : H×W×3 uint8   — source image in BGR for OpenCV
    """
    # Normalise aliked variant names for LightGlue's matcher key
    lg_key = "aliked" if features.startswith("aliked") else features

    extractor = _build_extractor(features, max_keypoints, device)
    matcher   = LightGlue(features=lg_key).eval().to(device)

    # load_image returns a (C, H, W) float32 RGB [0..1] tensor
    ref_tensor = load_image(ref_path).to(device)
    src_tensor = load_image(src_path).to(device)

    with torch.no_grad():
        feats_ref  = extractor.extract(ref_tensor)
        feats_src  = extractor.extract(src_tensor)
        matches_01 = matcher({"image0": feats_ref, "image1": feats_src})

    # Remove the batch dimension added by LightGlue
    feats_ref, feats_src, matches_01 = (
        rbd(feats_ref), rbd(feats_src), rbd(matches_01)
    )

    idx = matches_01["matches"]                                   # (M, 2)
    ref_pts = feats_ref["keypoints"][idx[:, 0]].cpu().numpy()    # (M, 2)
    src_pts = feats_src["keypoints"][idx[:, 1]].cpu().numpy()    # (M, 2)

    ref_bgr = cv2.imread(ref_path)
    src_bgr = cv2.imread(src_path)

    return ref_pts, src_pts, ref_bgr, src_bgr


# ===========================================================================
# 1b.  Homography pre-alignment (optional)
# ===========================================================================

def prealign_homography(
    ref_bgr: np.ndarray,
    src_bgr: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a RANSAC homography from the initial matches and warp the source.

    This corrects gross differences in scale, rotation, and perspective before
    the TPS pass, so LightGlue sees a well-aligned pair and returns many more
    high-confidence matches for the residual non-rigid step.

    Parameters
    ----------
    ref_bgr  : reference image (BGR uint8)
    src_bgr  : source image (BGR uint8)
    src_pts  : (N, 2) source keypoints from the initial match
    ref_pts  : (N, 2) reference keypoints from the initial match

    Returns
    -------
    src_prealigned : source image warped by the homography into the reference
                     coordinate frame (BGR uint8, same canvas size as ref_bgr)
    H              : 3×3 homography matrix (src → ref)
    inlier_mask    : boolean array of length N — True for RANSAC inliers
    """
    if len(src_pts) < 4:
        raise ValueError(
            f"Need ≥ 4 matches to compute a homography, got {len(src_pts)}."
        )

    H, mask = cv2.findHomography(
        src_pts.astype(np.float32),
        ref_pts.astype(np.float32),
        cv2.RANSAC,
        ransacReprojThreshold=8.0,
    )
    if H is None:
        raise RuntimeError("Homography estimation failed — too few inliers.")

    inlier_mask = mask.ravel().astype(bool)
    h, w = ref_bgr.shape[:2]
    src_prealigned = cv2.warpPerspective(src_bgr, H, (w, h))
    return src_prealigned, H, inlier_mask


def extract_and_match_array(
    ref_bgr: np.ndarray,
    src_bgr: np.ndarray,
    features: str = DEFAULT_FEATURES,
    device: torch.device = DEVICE,
    max_keypoints: int = MAX_KEYPOINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as extract_and_match but accepts in-memory BGR arrays instead of paths.

    Used for the second-pass match after homography pre-alignment.

    Returns
    -------
    ref_pts : (M, 2) float32
    src_pts : (M, 2) float32
    """
    lg_key    = "aliked" if features.startswith("aliked") else features
    extractor = _build_extractor(features, max_keypoints, device)
    matcher   = LightGlue(features=lg_key).eval().to(device)

    def bgr_to_tensor(img: np.ndarray) -> torch.Tensor:
        """Convert H×W×3 BGR uint8 → 3×H×W float32 RGB [0..1] on device."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t.to(device)

    ref_t = bgr_to_tensor(ref_bgr)
    src_t = bgr_to_tensor(src_bgr)

    with torch.no_grad():
        feats_ref  = extractor.extract(ref_t)
        feats_src  = extractor.extract(src_t)
        matches_01 = matcher({"image0": feats_ref, "image1": feats_src})

    feats_ref, feats_src, matches_01 = rbd(feats_ref), rbd(feats_src), rbd(matches_01)
    idx     = matches_01["matches"]
    ref_pts = feats_ref["keypoints"][idx[:, 0]].cpu().numpy()
    src_pts = feats_src["keypoints"][idx[:, 1]].cpu().numpy()
    return ref_pts, src_pts


def build_dilated_mask(
    mask_gray: np.ndarray,
    dilation_px: int = MASK_DILATION,
) -> np.ndarray:
    """Binarise and morphologically dilate a grayscale mask.

    Parameters
    ----------
    mask_gray   : H×W uint8 image (white pixels = structure of interest)
    dilation_px : dilation radius in pixels

    Returns
    -------
    Dilated binary mask (uint8, values 0 or 255).
    """
    _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    side   = 2 * dilation_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    return cv2.dilate(binary, kernel)


def filter_matches_by_mask(
    ref_pts: np.ndarray,
    src_pts: np.ndarray,
    ref_mask_dilated: np.ndarray | None = None,
    src_mask_dilated: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Discard matches whose keypoint falls inside either dilated mask.

    Providing a *reference* mask removes anchors that sit on the lesion in the
    baseline image.  Providing a *source* mask removes anchors that sit on the
    lesion in the moving image.  Both can be supplied simultaneously.

    Parameters
    ----------
    ref_pts           : (N, 2) reference keypoints
    src_pts           : (N, 2) source keypoints
    ref_mask_dilated  : dilated binary mask aligned to the reference image
    src_mask_dilated  : dilated binary mask aligned to the source image

    Returns
    -------
    Filtered ref_pts, src_pts with M ≤ N rows.
    """
    def outside(pts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return boolean array: True where the point is OUTSIDE the mask."""
        h, w = mask.shape[:2]
        result = np.ones(len(pts), dtype=bool)
        for i, (x, y) in enumerate(pts):
            xi = max(0, min(w - 1, int(round(x))))
            yi = max(0, min(h - 1, int(round(y))))
            result[i] = mask[yi, xi] == 0   # 0 = outside mask
        return result

    keep = np.ones(len(ref_pts), dtype=bool)
    if ref_mask_dilated is not None:
        keep &= outside(ref_pts, ref_mask_dilated)
    if src_mask_dilated is not None:
        keep &= outside(src_pts, src_mask_dilated)

    return ref_pts[keep], src_pts[keep]


# ===========================================================================
# 3.  Thin Plate Spline warping
# ===========================================================================

def warp_tps(
    src_bgr: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Compute a TPS warp that maps src_pts → ref_pts and applies it to src_bgr.

    OpenCV's createThinPlateSplineShapeTransformer computes a smooth,
    non-linear transformation driven by sparse control-point correspondences.
    It is well-suited to the slight, non-rigid deformations that occur when
    photographing the same body part over multiple sessions.

    Parameters
    ----------
    src_bgr     : H×W×3 source image (BGR uint8)
    src_pts     : (N, 2) source control points  (float32)
    ref_pts     : (N, 2) target control points in reference space (float32)
    output_size : desired (width, height) output; defaults to src_bgr size

    Returns
    -------
    Warped source image (BGR uint8).
    """
    if len(src_pts) < MIN_MATCHES:
        raise ValueError(
            f"Only {len(src_pts)} matches remain after filtering — need at "
            f"least {MIN_MATCHES} to compute a reliable TPS warp.  "
            "Try increasing max_keypoints or reducing the dilation radius."
        )

    tps = cv2.createThinPlateSplineShapeTransformer()

    # OpenCV expects shape (1, N, 2) and dtype float32
    src_cv = src_pts.astype(np.float32).reshape(1, -1, 2)
    ref_cv = ref_pts.astype(np.float32).reshape(1, -1, 2)

    # A trivial DMatch list: point i in src ↔ point i in ref
    cv_matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]

    # estimateTransformation(transformingShape, targetShape, matches)
    # The transformer will map transformingShape → targetShape when warpImage is called.
    tps.estimateTransformation(src_cv, ref_cv, cv_matches)

    if output_size is not None:
        out_w, out_h = output_size
        # Embed source in a canvas matching the reference dimensions so that
        # warpImage produces output in the reference coordinate frame.
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        sh, sw = src_bgr.shape[:2]
        canvas[: min(sh, out_h), : min(sw, out_w)] = src_bgr[
            : min(sh, out_h), : min(sw, out_w)
        ]
        return tps.warpImage(canvas)

    return tps.warpImage(src_bgr)


# ===========================================================================
# 4.  Visualisation
# ===========================================================================

def visualise(
    ref_bgr: np.ndarray,
    warped_bgr: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
    out_path: str | None = None,
) -> np.ndarray:
    """Produce a four-panel diagnostic image.

    Panels (left → right):
      1. Reference image
      2. Warped source image
      3. Alpha-blended overlay  (reference + warped)
      4. Absolute-difference heatmap (highlights residual misalignment)

    Parameters
    ----------
    ref_bgr    : reference image (BGR uint8)
    warped_bgr : warped source image (BGR uint8)
    alpha      : blend weight for the overlay panel
    out_path   : if given, save the composite to this path

    Returns
    -------
    Composite visualisation image (BGR uint8).
    """
    # Make sure both images share the same canvas size
    if warped_bgr.shape[:2] != ref_bgr.shape[:2]:
        warped_bgr = cv2.resize(
            warped_bgr, (ref_bgr.shape[1], ref_bgr.shape[0])
        )

    overlay  = cv2.addWeighted(ref_bgr, 1.0 - alpha, warped_bgr, alpha, 0)
    diff     = cv2.absdiff(ref_bgr, warped_bgr)
    diff_map = cv2.applyColorMap(
        cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET
    )

    composite = np.concatenate([ref_bgr, warped_bgr, overlay, diff_map], axis=1)

    labels = ["Reference", "Warped Source", f"Overlay (α={alpha:.1f})", "Abs-diff"]
    panel_w = ref_bgr.shape[1]
    for col, label in enumerate(labels):
        cv2.putText(
            composite, label,
            (col * panel_w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
            cv2.LINE_AA,
        )

    if out_path:
        cv2.imwrite(out_path, composite)
        print(f"  Visualisation saved → {out_path}")

    return composite


# ===========================================================================
# Top-level pipeline
# ===========================================================================

def register(
    ref_path: str,
    src_path: str,
    features: str = DEFAULT_FEATURES,
    max_keypoints: int = MAX_KEYPOINTS,
    use_homography: bool = True,
    mask_ref_path: str | None = None,
    mask_src_path: str | None = None,
    dilation_px: int = MASK_DILATION,
    out_warped: str = "warped.png",
    out_vis: str = "visualisation.png",
) -> np.ndarray:
    """Full lesion-aware registration pipeline.

    Parameters
    ----------
    ref_path        : path to the reference (fixed / baseline) image
    src_path        : path to the source (moving) image
    features        : feature extractor (superpoint / disk / aliked-n16 / aliked-n32)
    max_keypoints   : max keypoints per image
    use_homography  : if True, compute a RANSAC homography from the initial
                      matches to pre-align the source, then re-match for TPS
    mask_ref_path   : binary lesion mask for the REFERENCE image (white = lesion)
    mask_src_path   : binary lesion mask for the SOURCE image (white = lesion)
    dilation_px     : pixels to dilate the mask exclusion zone
    out_warped      : file path for the warped source image
    out_vis         : file path for the visualisation composite

    Returns
    -------
    Warped source image (BGR uint8).
    """
    # ------------------------------------------------------------------
    # Step 1 — Initial feature matching
    # ------------------------------------------------------------------
    print(f"[1/5] Extracting {features} features (max {max_keypoints}) and matching with LightGlue …")
    ref_pts_init, src_pts_init, ref_bgr, src_bgr = extract_and_match(
        ref_path, src_path, features=features, max_keypoints=max_keypoints
    )
    print(f"      {len(src_pts_init)} raw matches found.")

    # ------------------------------------------------------------------
    # Step 1b — Homography pre-alignment (optional)
    # ------------------------------------------------------------------
    if use_homography:
        print("[2/5] Computing RANSAC homography for global pre-alignment …")
        try:
            src_prealigned, H, inliers = prealign_homography(
                ref_bgr, src_bgr, src_pts_init, ref_pts_init
            )
            print(f"      Homography estimated ({inliers.sum()} / {len(inliers)} inliers).")
            print("      Re-matching on pre-aligned image pair …")
            ref_pts, src_pts_aligned = extract_and_match_array(
                ref_bgr, src_prealigned,
                features=features, max_keypoints=max_keypoints,
            )
            # src_pts_aligned are in the homography-warped space; that IS the
            # source space we will warp from, so use src_prealigned as src_bgr.
            src_pts = src_pts_aligned
            src_bgr = src_prealigned
            print(f"      {len(src_pts)} matches after pre-alignment re-match.")
        except (ValueError, RuntimeError) as exc:
            print(f"      WARNING: homography step failed ({exc}) — falling back to direct TPS.")
            ref_pts, src_pts = ref_pts_init, src_pts_init
    else:
        print("[2/5] Skipping homography pre-alignment (--no-homography).")
        ref_pts, src_pts = ref_pts_init, src_pts_init

    # ------------------------------------------------------------------
    # Step 3 — Mask-based filtering
    # ------------------------------------------------------------------
    ref_dilated = src_dilated = None
    if mask_ref_path is not None or mask_src_path is not None:
        print(f"[3/5] Loading lesion mask(s) and dilating by {dilation_px} px …")
        if mask_ref_path is not None:
            g = cv2.imread(mask_ref_path, cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise FileNotFoundError(f"Reference mask not found: {mask_ref_path}")
            ref_dilated = build_dilated_mask(g, dilation_px)
        if mask_src_path is not None:
            g = cv2.imread(mask_src_path, cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise FileNotFoundError(f"Source mask not found: {mask_src_path}")
            src_dilated = build_dilated_mask(g, dilation_px)
        ref_pts, src_pts = filter_matches_by_mask(
            ref_pts, src_pts,
            ref_mask_dilated=ref_dilated,
            src_mask_dilated=src_dilated,
        )
        print(f"      {len(src_pts)} matches remain after mask filtering.")
    else:
        print("[3/5] No mask provided — skipping lesion filtering.")

    # ------------------------------------------------------------------
    # Step 4 — TPS warp
    # ------------------------------------------------------------------
    print("[4/5] Computing Thin Plate Spline transformation and warping …")
    output_size = (ref_bgr.shape[1], ref_bgr.shape[0])   # (W, H) of reference
    warped = warp_tps(src_bgr, src_pts, ref_pts, output_size=output_size)
    cv2.imwrite(out_warped, warped)
    print(f"      Warped image saved → {out_warped}")

    # ------------------------------------------------------------------
    # Step 5 — Visualisation
    # ------------------------------------------------------------------
    print("[5/5] Generating diagnostic visualisation …")
    ref_bgr_orig = cv2.imread(ref_path)
    visualise(ref_bgr_orig, warped, out_path=out_vis)

    print("Done.")
    return warped


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Lesion-aware non-rigid image registration (SuperPoint + LightGlue + TPS).\n"
            "Aligns SOURCE to REFERENCE while preserving the lesion's relative size."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("reference",      help="Path to the reference image.")
    parser.add_argument("source",         help="Path to the source (moving) image.")
    parser.add_argument(
        "--mask-ref", default=None,
        metavar="PATH",
        help="Binary lesion mask for the REFERENCE (baseline) image (white = lesion).",
    )
    parser.add_argument(
        "--mask-src", default=None,
        metavar="PATH",
        help="Binary lesion mask for the SOURCE (moving) image (white = lesion).",
    )
    parser.add_argument(
        "--dilation", type=int, default=MASK_DILATION,
        metavar="PX",
        help=f"Mask dilation radius in pixels (default: {MASK_DILATION}).",
    )
    parser.add_argument(
        "--no-homography", action="store_true", default=False,
        help="Disable the RANSAC homography pre-alignment step (enabled by default).",
    )
    parser.add_argument(
        "--features", default=DEFAULT_FEATURES, choices=FEATURES_CHOICES,
        help=(
            f"Feature extractor to use (default: {DEFAULT_FEATURES}). "
            "disk and aliked-n32 are most robust on skin imagery."
        ),
    )
    parser.add_argument(
        "--max-keypoints", type=int, default=MAX_KEYPOINTS,
        metavar="N",
        help=f"Max keypoints per image (default: {MAX_KEYPOINTS}). Increase for more matches.",
    )
    parser.add_argument(
        "--out-warped", default="warped.png",
        metavar="PATH",
        help="Output path for the warped source image (default: warped.png).",
    )
    parser.add_argument(
        "--out-vis", default="visualisation.png",
        metavar="PATH",
        help="Output path for the visualisation composite (default: visualisation.png).",
    )
    args = parser.parse_args()

    register(
        ref_path=args.reference,
        src_path=args.source,
        features=args.features,
        max_keypoints=args.max_keypoints,
        use_homography=not args.no_homography,
        mask_ref_path=args.mask_ref,
        mask_src_path=args.mask_src,
        dilation_px=args.dilation,
        out_warped=args.out_warped,
        out_vis=args.out_vis,
    )
