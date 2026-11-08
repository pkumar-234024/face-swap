from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.eye_sync import EyeSynchronizer
from modules.face_detector import FaceDetector
from modules.facial_landmarks import FacialLandmarks, FacialLandmarkTracker
from modules.head_pose import HeadPoseEstimator
from modules.lip_sync import LipSynchronizer


def test_face_detector_initialization():
    detector = FaceDetector(device="cpu")
    assert detector is not None


def test_face_detection_on_synthetic_frame():
    detector = FaceDetector(device="cpu")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detector.detect_faces(blank)
    assert isinstance(faces, list)
    best = detector.detect_best_face(blank)
    assert best is None or isinstance(best, dict)


def test_face_detection_on_sample_asset():
    asset_path = Path("assets/g1.png")
    if not asset_path.exists():
        pytest.skip("assets/g1.png not found")

    img = cv2.imread(str(asset_path))
    assert img is not None

    detector = FaceDetector(device="cpu")
    best_face = detector.detect_best_face(img)
    assert best_face is not None
    assert "bbox" in best_face
    assert len(best_face["bbox"]) == 4
    assert best_face["det_score"] > 0.5


def test_facial_landmark_tracker():
    tracker = FacialLandmarkTracker(smoothing=True)
    # Synthetic 68 points
    synthetic_pts = np.random.uniform(50, 200, size=(68, 2)).astype(np.float32)
    face_data = {
        "landmark_3d_68": synthetic_pts,
        "bbox": np.array([50, 50, 200, 200], dtype=np.float32),
        "det_score": 0.95,
    }
    landmarks = tracker.extract_landmarks(face_data, (480, 640))
    assert landmarks is not None
    assert landmarks.points_68.shape[0] == 68
    assert len(landmarks.jaw) == 17
    assert len(landmarks.right_eye) == 6
    assert len(landmarks.left_eye) == 6
    assert len(landmarks.outer_lips) == 12


def test_eye_synchronizer():
    sync = EyeSynchronizer()
    # Canonical open eye points: 6 points
    open_eye = np.array([
        [10.0, 50.0],  # 0
        [20.0, 40.0],  # 1
        [30.0, 40.0],  # 2
        [40.0, 50.0],  # 3
        [30.0, 60.0],  # 4
        [20.0, 60.0],  # 5
    ], dtype=np.float32)
    ear_open = sync.compute_ear(open_eye)
    assert ear_open > 0.25

    # Closed eye points
    closed_eye = np.array([
        [10.0, 50.0],
        [20.0, 49.0],
        [30.0, 49.0],
        [40.0, 50.0],
        [30.0, 51.0],
        [20.0, 51.0],
    ], dtype=np.float32)
    ear_closed = sync.compute_ear(closed_eye)
    assert ear_closed < ear_open


def test_lip_synchronizer():
    sync = LipSynchronizer()
    pts_68 = np.ones((68, 2), dtype=np.float32) * 100.0
    # Set mouth corners
    pts_68[48] = [80.0, 100.0]
    pts_68[54] = [120.0, 100.0]
    # Set upper and lower lips
    pts_68[51] = [100.0, 95.0]
    pts_68[57] = [100.0, 105.0]
    pts_68[33] = [100.0, 80.0]
    pts_68[8] = [100.0, 130.0]

    state = sync.analyze_mouth(pts_68, face_size=100.0)
    assert state.mouth_aspect_ratio >= 0.0
    assert 0.0 <= state.smile_intensity <= 1.0


def test_head_pose_estimator():
    estimator = HeadPoseEstimator()
    pts_68 = np.zeros((68, 2), dtype=np.float32)
    # Populate key points
    pts_68[30] = [320, 240]  # nose tip
    pts_68[8] = [320, 360]   # chin
    pts_68[45] = [260, 200]  # left eye corner
    pts_68[36] = [380, 200]  # right eye corner
    pts_68[54] = [280, 300]  # left mouth
    pts_68[48] = [360, 300]  # right mouth

    pose = estimator.estimate_pose(pts_68, (480, 640))
    assert isinstance(pose.pitch, float)
    assert isinstance(pose.yaw, float)
    assert isinstance(pose.roll, float)


def test_face_blender_modes_and_teeth_preservation():
    from modules.face_blending import FaceBlender

    blender = FaceBlender(temporal_smoothing=False)
    target_frame = np.full((200, 200, 3), 120, dtype=np.uint8)
    # Bright white square representing teeth in mouth area
    target_frame[90:110, 90:110] = [255, 255, 255]

    warped_roi = np.full((120, 120, 3), 40, dtype=np.uint8)
    roi_mask = np.full((120, 120), 255, dtype=np.uint8)
    roi_bbox = (40, 40, 160, 160)

    # 68 landmarks in ROI space with open mouth (points 60-67)
    lm_roi = np.full((68, 2), 60.0, dtype=np.float32)
    lm_roi[60] = [45.0, 60.0]
    lm_roi[64] = [75.0, 60.0]
    lm_roi[62] = [60.0, 50.0]
    lm_roi[66] = [60.0, 70.0]
    # Set other inner points
    lm_roi[61] = [52.0, 53.0]
    lm_roi[63] = [68.0, 53.0]
    lm_roi[65] = [68.0, 67.0]
    lm_roi[67] = [52.0, 67.0]

    # Test multiband blend
    blended_mb = blender.blend_face(
        target_frame=target_frame,
        warped_roi=warped_roi,
        roi_mask=roi_mask,
        roi_bbox=roi_bbox,
        target_landmarks_roi=lm_roi,
        preserve_mouth_interior=True,
        enhance_teeth=True,
        method="multiband",
    )
    assert blended_mb.shape == (200, 200, 3)
    # The center of the mouth aperture should preserve high pixel intensity from target_frame teeth
    assert blended_mb[100, 100, 0] > 180

    # Test poisson blend
    blended_poisson = blender.blend_face(
        target_frame=target_frame,
        warped_roi=warped_roi,
        roi_mask=roi_mask,
        roi_bbox=roi_bbox,
        target_landmarks_roi=lm_roi,
        preserve_mouth_interior=True,
        enhance_teeth=False,
        method="poisson",
    )
    assert blended_poisson.shape == (200, 200, 3)


