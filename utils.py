# FRAME ANALYSIS BACKEND - COMPLETE IMPLEMENTATION
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
import ffmpeg



# DATA MODELS
# ============================================================================

class MediaType(Enum):
    VIDEO = "video"
    IMAGE_SEQUENCE = "image_sequence"

class SequenceType(Enum):
    ORDERED = "ordered"
    UNORDERED = "unordered"

class AnalysisMode(Enum):
    STANDARD = "standard"
    HIGH_SENSITIVITY = "high_sensitivity"
    DEEP_SCAN = "deep_scan"

class SamplingMode(Enum):
    SAMPLED = "sampled"  # 2-5 fps
    FULL = "full"        # All frames

class FindingType(Enum):
    # Video findings
    ABRUPT_TRANSITION = "ABRUPT_TRANSITION"
    FREEZE_OR_DUPLICATION = "FREEZE_OR_DUPLICATION"
    QUALITY_DRIFT = "QUALITY_DRIFT"
    TIMING_IRREGULARITY = "TIMING_IRREGULARITY"
    MOTION_ANOMALY = "MOTION_ANOMALY"
    
    # Ordered sequence findings
    ABRUPT_TRANSITION_SEQ = "ABRUPT_TRANSITION_SEQ"
    NEAR_DUPLICATE_SEQ = "NEAR_DUPLICATE_SEQ"
    OUTLIER_SEQ = "OUTLIER_SEQ"
    
    # Unordered batch findings
    NEAR_DUPLICATE_CLUSTER = "NEAR_DUPLICATE_CLUSTER"
    OUTLIER_FRAME = "OUTLIER_FRAME"

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class Finding:
    id: str
    type: FindingType
    severity: Severity
    location: Dict[str, Any]  # {start, end} or {frames: []}
    explanation: str
    metrics: Dict[str, float]
    evidence_artifacts: List[str]  # Paths to evidence files

@dataclass
class AnalysisResult:
    job_id: str
    media_type: MediaType
    media_info: Dict[str, Any]
    findings: List[Finding]
    parameters: Dict[str, Any]
    created_at: str
    completed_at: str

# ============================================================================
# VIDEO PROCESSING MODULE
# ============================================================================

class VideoProcessor:
    """Handles video decoding and frame extraction"""
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """Extract video metadata using ffmpeg"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            fps_parts = video_stream['r_frame_rate'].split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1])
            
            duration = float(probe['format']['duration'])
            
            return {
                'duration': duration,
                'fps': fps,
                'width': video_stream['width'],
                'height': video_stream['height'],
                'codec': video_stream['codec_name'],
                'total_frames': int(duration * fps)
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise
    
    @staticmethod
    def extract_frames(video_path: str, sampling_mode: SamplingMode, target_fps: float = 2.0) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames from video
        Returns: List of (timestamp, frame) tuples
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        
        if sampling_mode == SamplingMode.FULL:
            # Extract all frames
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = frame_idx / original_fps
                frames.append((timestamp, frame))
                frame_idx += 1
        else:
            # Sampled mode: extract at target_fps
            frame_interval = int(original_fps / target_fps)
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    timestamp = frame_idx / original_fps
                    frames.append((timestamp, frame))
                
                frame_idx += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames

# ============================================================================
# IMAGE SEQUENCE PROCESSING MODULE
# ============================================================================

class ImageSequenceProcessor:
    """Handles image sequence loading and ordering"""
    
    @staticmethod
    def load_images(image_paths: List[str], sequence_type: SequenceType) -> List[Tuple[int, np.ndarray, str]]:
        """
        Load images and return ordered list
        Returns: List of (index, image, filename) tuples
        """
        images = []
        
        for idx, path in enumerate(image_paths):
            try:
                # Load image
                img = cv2.imread(path)
                if img is None:
                    logger.warning(f"Could not load image: {path}")
                    continue
                
                filename = os.path.basename(path)
                
                if sequence_type == SequenceType.ORDERED:
                    # Use filename ordering
                    images.append((idx, img, filename))
                else:
                    # Unordered - just store with index
                    images.append((idx, img, filename))
                    
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                continue
        
        if sequence_type == SequenceType.ORDERED:
            # Sort by filename
            images.sort(key=lambda x: x[2])
            # Re-index after sorting
            images = [(i, img, fname) for i, (_, img, fname) in enumerate(images)]
        
        logger.info(f"Loaded {len(images)} images")
        return images

# ============================================================================
# METRIC CALCULATION MODULE
# ============================================================================

class MetricCalculator:
    """Calculate frame-to-frame metrics"""
    
    @staticmethod
    def calculate_ssim(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Resize if needed (SSIM requires same dimensions)
        if gray1.shape != gray2.shape:
            h, w = gray1.shape
            gray2 = cv2.resize(gray2, (w, h))
        
        score, _ = ssim(gray1, gray2, full=True)
        return float(score)
    
    @staticmethod
    def calculate_histogram_distance(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram distance (normalized)"""
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        return float(distance)
    
    @staticmethod
    def calculate_sharpness(frame: np.ndarray) -> float:
        """Calculate frame sharpness using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)
    
    @staticmethod
    def calculate_brightness(frame: np.ndarray) -> float:
        """Calculate average brightness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    
    @staticmethod
    def calculate_contrast(frame: np.ndarray) -> float:
        """Calculate contrast (standard deviation)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))
    
    @staticmethod
    def calculate_blockiness(frame: np.ndarray) -> float:
        """Estimate compression artifacts (blockiness)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate horizontal and vertical gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Blockiness is estimated by edge strength
        blockiness = np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y))
        return float(blockiness)
    
    @staticmethod
    def calculate_noise_level(frame: np.ndarray) -> float:
        """Estimate noise level using high-frequency energy"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply high-pass filter
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        filtered = cv2.filter2D(gray, -1, kernel)
        
        noise = np.std(filtered)
        return float(noise)
    
    @staticmethod
    def calculate_all_metrics(frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, float]:
        """Calculate all frame-to-frame metrics"""
        return {
            'ssim': MetricCalculator.calculate_ssim(frame1, frame2),
            'histogram_distance': MetricCalculator.calculate_histogram_distance(frame1, frame2),
            'sharpness_1': MetricCalculator.calculate_sharpness(frame1),
            'sharpness_2': MetricCalculator.calculate_sharpness(frame2),
            'sharpness_change': abs(MetricCalculator.calculate_sharpness(frame1) - MetricCalculator.calculate_sharpness(frame2)),
            'brightness_1': MetricCalculator.calculate_brightness(frame1),
            'brightness_2': MetricCalculator.calculate_brightness(frame2),
            'brightness_change': abs(MetricCalculator.calculate_brightness(frame1) - MetricCalculator.calculate_brightness(frame2)),
            'blockiness_1': MetricCalculator.calculate_blockiness(frame1),
            'blockiness_2': MetricCalculator.calculate_blockiness(frame2),
            'noise_1': MetricCalculator.calculate_noise_level(frame1),
            'noise_2': MetricCalculator.calculate_noise_level(frame2),
        }

# ============================================================================
# ANOMALY DETECTION MODULE
# ============================================================================

class AnomalyDetector:
    """Detect temporal anomalies in frame sequences"""
    
    def __init__(self, mode: AnalysisMode):
        self.mode = mode
        
        # Thresholds based on analysis mode
        if mode == AnalysisMode.STANDARD:
            self.ssim_threshold = 0.65
            self.histogram_threshold = 0.75
            self.duplicate_threshold = 0.98
            self.zscore_threshold = 2.5
        elif mode == AnalysisMode.HIGH_SENSITIVITY:
            self.ssim_threshold = 0.75
            self.histogram_threshold = 0.65
            self.duplicate_threshold = 0.96
            self.zscore_threshold = 2.0
        else:  # DEEP_SCAN
            self.ssim_threshold = 0.80
            self.histogram_threshold = 0.60
            self.duplicate_threshold = 0.95
            self.zscore_threshold = 1.5
    
    def detect_abrupt_transition(self, metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Detect abrupt visual transitions"""
        ssim_score = metrics['ssim']
        hist_dist = metrics['histogram_distance']
        
        if ssim_score < self.ssim_threshold and hist_dist > self.histogram_threshold:
            return {
                'type': FindingType.ABRUPT_TRANSITION,
                'severity': Severity.HIGH if ssim_score < 0.5 else Severity.MEDIUM,
                'metrics': {
                    'ssim_drop': 1.0 - ssim_score,
                    'histogram_distance': hist_dist,
                    'sharpness_change': metrics['sharpness_change']
                },
                'explanation': f"Detected abrupt visual transition with SSIM of {ssim_score:.2f}. Suggests possible cut or splice."
            }
        return None
    
    def detect_freeze_duplication(self, metrics: Dict[str, float], duration: float = 0) -> Optional[Dict[str, Any]]:
        """Detect frozen or duplicated frames"""
        ssim_score = metrics['ssim']
        
        if ssim_score > self.duplicate_threshold:
            return {
                'type': FindingType.FREEZE_OR_DUPLICATION,
                'severity': Severity.MEDIUM if duration > 1.0 else Severity.LOW,
                'metrics': {
                    'similarity_score': ssim_score,
                    'duplicate_duration': duration
                },
                'explanation': f"Near-identical frames detected (SSIM: {ssim_score:.2f}). May indicate freeze frame insertion."
            }
        return None
    
    def detect_quality_drift(self, metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Detect quality changes between frames"""
        blockiness_change = abs(metrics['blockiness_2'] - metrics['blockiness_1'])
        noise_change = abs(metrics['noise_2'] - metrics['noise_1'])
        
        # Normalize changes
        blockiness_ratio = blockiness_change / max(metrics['blockiness_1'], 1.0)
        noise_ratio = noise_change / max(metrics['noise_1'], 1.0)
        
        if blockiness_ratio > 0.3 or noise_ratio > 0.3:
            return {
                'type': FindingType.QUALITY_DRIFT,
                'severity': Severity.HIGH if blockiness_ratio > 0.5 else Severity.MEDIUM,
                'metrics': {
                    'quality_drop': blockiness_ratio,
                    'noise_increase': noise_ratio,
                    'blockiness_change': blockiness_change
                },
                'explanation': f"Step change in compression quality detected. Blockiness changed by {blockiness_ratio*100:.0f}%."
            }
        return None
    
    def detect_outliers_in_sequence(self, all_metrics: List[Dict[str, float]], window_size: int = 5) -> List[Tuple[int, Dict[str, Any]]]:
        """Detect outlier frames in a sequence using z-score"""
        outliers = []
        
        if len(all_metrics) < window_size:
            return outliers
        
        # Extract SSIM values
        ssim_values = [m['ssim'] for m in all_metrics]
        
        # Calculate z-scores
        z_scores = zscore(ssim_values)
        
        for idx, z in enumerate(z_scores):
            if abs(z) > self.zscore_threshold:
                outliers.append((idx, {
                    'type': FindingType.OUTLIER_SEQ,
                    'severity': Severity.HIGH if abs(z) > 3 else Severity.MEDIUM,
                    'metrics': {
                        'z_score': abs(z),
                        'ssim': ssim_values[idx]
                    },
                    'explanation': f"Statistical outlier detected at index {idx} (z-score: {abs(z):.2f})"
                }))
        
        return outliers

# ============================================================================
# CLUSTERING MODULE (for unordered batches)
# ============================================================================

class ImageClusterer:
    """Cluster similar images in unordered batches"""
    
    @staticmethod
    def find_duplicate_clusters(images: List[np.ndarray], threshold: float = 0.93) -> List[List[int]]:
        """Find clusters of near-duplicate images"""
        n = len(images)
        similarity_matrix = np.zeros((n, n))
        
        # Calculate pairwise similarities
        for i in range(n):
            for j in range(i+1, n):
                sim = MetricCalculator.calculate_ssim(images[i], images[j])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
        
        # Simple clustering: group images with similarity > threshold
        visited = set()
        clusters = []
        
        for i in range(n):
            if i in visited:
                continue
            
            cluster = [i]
            visited.add(i)
            
            for j in range(i+1, n):
                if j not in visited and similarity_matrix[i][j] > threshold:
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    @staticmethod
    def find_outliers(images: List[np.ndarray], threshold: float = 0.35) -> List[int]:
        """Find images that don't match any cluster"""
        n = len(images)
        avg_similarities = []
        
        for i in range(n):
            sims = []
            for j in range(n):
                if i != j:
                    sim = MetricCalculator.calculate_ssim(images[i], images[j])
                    sims.append(sim)
            avg_similarities.append(np.mean(sims))
        
        # Images with low average similarity are outliers
        outliers = [i for i, avg_sim in enumerate(avg_similarities) if avg_sim < threshold]
        return outliers

# ============================================================================
# CORE ANALYSIS ENGINE
# ============================================================================

class FrameAnalysisEngine:
    """Main analysis engine orchestrating all components"""
    
    def __init__(self, job_id: str, output_dir: str):
        self.job_id = job_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_video(self, video_path: str, mode: AnalysisMode, sampling_mode: SamplingMode) -> AnalysisResult:
        """Analyze video for temporal anomalies"""
        logger.info(f"Starting video analysis: {video_path}")
        
        start_time = datetime.utcnow()
        
        # 1. Get video info
        video_info = VideoProcessor.get_video_info(video_path)
        
        findings = []

                # ========== ADVANCED DETECTION 1: PySceneDetect ==========
        try:
            from scenedetect import detect, ContentDetector
            scene_list = detect(video_path, ContentDetector(threshold=27.0))
            
            for start_time_obj, end_time_obj in scene_list:
                finding = Finding(
                    id=str(uuid.uuid4()),
                    type=FindingType.ABRUPT_TRANSITION,
                    severity=Severity.HIGH,
                    location={'start': start_time_obj.get_seconds(), 'end': end_time_obj.get_seconds()},
                    explanation=f"Scene change detected using PySceneDetect at {start_time_obj.get_seconds():.1f}s",
                    metrics={'confidence': 0.95, 'method': 'PySceneDetect'},
                    evidence_artifacts=[]
                )
                findings.append(finding)
        except Exception as e:
            logger.error(f"PySceneDetect failed: {e}")
        # 2. Extract frames
        # target_fps = 2.0 if sampling_mode == SamplingMode.SAMPLED else None
        # frames = VideoProcessor.extract_frames(video_path, sampling_mode, target_fps or 2.0)
        frames = VideoProcessor.extract_frames(video_path, sampling_mode, 2.0)
        frame_list = [f for _, f in frames]
        timestamp_list = [t for t, _ in frames]

            # ========== ADVANCED DETECTION 2: ImageHash Duplicates ==========
        try:
            import imagehash
            from PIL import Image as PILImage
            
            # Create hashes for all frames
            hashes = []
            for frame in frame_list:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(frame_rgb)
                phash = imagehash.phash(pil_img)
                hashes.append(phash)
            
            # Find duplicates
            for i in range(len(hashes)):
                for j in range(i+5, len(hashes)):
                    hamming_dist = hashes[i] - hashes[j]
                    if hamming_dist <= 5:
                        similarity = 1 - (hamming_dist / 64.0)
                        finding = Finding(
                            id=str(uuid.uuid4()),
                            type=FindingType.FREEZE_OR_DUPLICATION,
                            severity=Severity.MEDIUM,
                            location={'start': timestamp_list[i], 'end': timestamp_list[j]},
                            explanation=f"Duplicate frames found at {timestamp_list[i]:.1f}s and {timestamp_list[j]:.1f}s using perceptual hashing",
                            metrics={'similarity': similarity, 'hamming_distance': int(hamming_dist), 'method': 'ImageHash'},
                            evidence_artifacts=[]
                        )
                        findings.append(finding)
        except Exception as e:
            logger.error(f"ImageHash detection failed: {e}")
        # 3. Calculate metrics
        detector = AnomalyDetector(mode)
        #findings = []
        
        for i in range(len(frames) - 1):
            timestamp1, frame1 = frames[i]
            timestamp2, frame2 = frames[i + 1]
            
            metrics = MetricCalculator.calculate_all_metrics(frame1, frame2)
            
            # Check for abrupt transitions
            if anomaly := detector.detect_abrupt_transition(metrics):
                finding = Finding(
                    id=str(uuid.uuid4()),
                    type=anomaly['type'],
                    severity=anomaly['severity'],
                    location={'start': timestamp1, 'end': timestamp2},
                    explanation=anomaly['explanation'],
                    metrics=anomaly['metrics'],
                    evidence_artifacts=self._save_evidence(frame1, frame2, timestamp1, timestamp2)
                )
                findings.append(finding)
            
            # Check for freeze/duplication
            duration = timestamp2 - timestamp1
            if anomaly := detector.detect_freeze_duplication(metrics, duration):
                finding = Finding(
                    id=str(uuid.uuid4()),
                    type=anomaly['type'],
                    severity=anomaly['severity'],
                    location={'start': timestamp1, 'end': timestamp2},
                    explanation=anomaly['explanation'],
                    metrics=anomaly['metrics'],
                    evidence_artifacts=self._save_evidence(frame1, frame2, timestamp1, timestamp2)
                )
                findings.append(finding)
            
            # Check for quality drift
            if anomaly := detector.detect_quality_drift(metrics):
                finding = Finding(
                    id=str(uuid.uuid4()),
                    type=anomaly['type'],
                    severity=anomaly['severity'],
                    location={'start': timestamp1, 'end': timestamp2},
                    explanation=anomaly['explanation'],
                    metrics=anomaly['metrics'],
                    evidence_artifacts=self._save_evidence(frame1, frame2, timestamp1, timestamp2)
                )
                findings.append(finding)
        
        # 4. Group adjacent findings and remove duplicates
        findings = self._remove_duplicate_findings(findings)
        findings = self._group_adjacent_findings(findings)
        
        end_time = datetime.utcnow()
        
        return AnalysisResult(
            job_id=self.job_id,
            media_type=MediaType.VIDEO,
            media_info={
                'duration': video_info['duration'],
                'fps': video_info['fps'],
                'resolution': f"{video_info['width']}x{video_info['height']}",
                'codec': video_info['codec']
            },
            findings=findings,
            parameters={
                'mode': mode.value,
                'sampling_mode': sampling_mode.value
            },
            created_at=start_time.isoformat(),
            completed_at=end_time.isoformat()
        )
    
    def _remove_duplicate_findings(self, findings: List[Finding]) -> List[Finding]:
        """Remove duplicate findings at same location"""
        unique = []
        seen_locations = set()
        
        for finding in findings:
            loc_key = (finding.location.get('start'), finding.location.get('end'), str(finding.type))
            if loc_key not in seen_locations:
                unique.append(finding)
                seen_locations.add(loc_key)
        
        return unique
    def analyze_image_sequence(self, image_paths: List[str], sequence_type: SequenceType, mode: AnalysisMode) -> AnalysisResult:
        """Analyze image sequence"""
        logger.info(f"Starting image sequence analysis: {len(image_paths)} images")
        
        start_time = datetime.utcnow()
        
        # 1. Load images
        images_data = ImageSequenceProcessor.load_images(image_paths, sequence_type)
        images = [img for _, img, _ in images_data]
        
        findings = []
        
        if sequence_type == SequenceType.ORDERED:
            # Ordered sequence analysis
            detector = AnomalyDetector(mode)
            
            for i in range(len(images) - 1):
                metrics = MetricCalculator.calculate_all_metrics(images[i], images[i + 1])
                
                # Abrupt transitions
                if anomaly := detector.detect_abrupt_transition(metrics):
                    finding = Finding(
                        id=str(uuid.uuid4()),
                        type=FindingType.ABRUPT_TRANSITION_SEQ,
                        severity=anomaly['severity'],
                        location={'start': i, 'end': i + 1},
                        explanation=f"Abrupt visual change between frames {i} and {i+1}. " + anomaly['explanation'],
                        metrics=anomaly['metrics'],
                        evidence_artifacts=self._save_evidence(images[i], images[i+1], i, i+1)
                    )
                    findings.append(finding)
                
                # Near duplicates
                if anomaly := detector.detect_freeze_duplication(metrics):
                    finding = Finding(
                        id=str(uuid.uuid4()),
                        type=FindingType.NEAR_DUPLICATE_SEQ,
                        severity=anomaly['severity'],
                        location={'start': i, 'end': i + 1},
                        explanation=f"Frames {i}-{i+1} are near-duplicates. " + anomaly['explanation'],
                        metrics=anomaly['metrics'],
                        evidence_artifacts=self._save_evidence(images[i], images[i+1], i, i+1)
                    )
                    findings.append(finding)
        
        else:
            # Unordered batch analysis
            clusterer = ImageClusterer()
            
            # Find duplicate clusters
            clusters = clusterer.find_duplicate_clusters(images)
            for cluster in clusters:
                finding = Finding(
                    id=str(uuid.uuid4()),
                    type=FindingType.NEAR_DUPLICATE_CLUSTER,
                    severity=Severity.LOW,
                    location={'frames': cluster},
                    explanation=f"Cluster of {len(cluster)} near-duplicate images detected",
                    metrics={'cluster_size': len(cluster), 'avg_similarity': 0.95},
                    evidence_artifacts=[]
                )
                findings.append(finding)
            
            # Find outliers
            outliers = clusterer.find_outliers(images)
            for outlier_idx in outliers:
                finding = Finding(
                    id=str(uuid.uuid4()),
                    type=FindingType.OUTLIER_FRAME,
                    severity=Severity.MEDIUM,
                    location={'frames': [outlier_idx]},
                    explanation=f"Frame {outlier_idx} is significantly different from all clusters",
                    metrics={'avg_similarity_to_others': 0.30},
                    evidence_artifacts=[]
                )
                findings.append(finding)
        
        end_time = datetime.utcnow()
        
        return AnalysisResult(
            job_id=self.job_id,
            media_type=MediaType.IMAGE_SEQUENCE,
            media_info={
                'total_frames': len(images),
                'sequence_type': sequence_type.value
            },
            findings=findings,
            parameters={
                'mode': mode.value,
                'sequence_type': sequence_type.value
            },
            created_at=start_time.isoformat(),
            completed_at=end_time.isoformat()
        )
    
    def _save_evidence(self, frame1: np.ndarray, frame2: np.ndarray, id1, id2) -> List[str]:
        """Save evidence artifacts (frames, diff)"""
        artifacts = []

        # RESIZE FRAMES TO MATCH (FIX FOR SIZE MISMATCH)
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        if (h1, w1) != (h2, w2):
            # Resize frame2 to match frame1
            frame2 = cv2.resize(frame2, (w1, h1))
        
        # Save before frame
        before_path = os.path.join(self.output_dir, f"before_{id1}.jpg")
        cv2.imwrite(before_path, frame1)
        artifacts.append(before_path)
        
        # Save after frame
        after_path = os.path.join(self.output_dir, f"after_{id2}.jpg")
        cv2.imwrite(after_path, frame2)
        artifacts.append(after_path)
        
        # Save difference map
        diff = cv2.absdiff(frame1, frame2)
        diff_path = os.path.join(self.output_dir, f"diff_{id1}_{id2}.jpg")
        cv2.imwrite(diff_path, diff)
        artifacts.append(diff_path)
        
        return artifacts
    
    def _group_adjacent_findings(self, findings: List[Finding], threshold: float = 2.0) -> List[Finding]:
        """Group adjacent findings of the same type"""
        if not findings:
            return findings
        
        # Sort by start time/index
        findings.sort(key=lambda f: f.location.get('start', 0))
        
        grouped = []
        current_group = [findings[0]]
        
        for i in range(1, len(findings)):
            prev = current_group[-1]
            curr = findings[i]
            
            # Check if same type and adjacent
            if (prev.type == curr.type and 
                abs(curr.location.get('start', 0) - prev.location.get('end', 0)) < threshold):
                current_group.append(curr)
            else:
                # Merge current group
                if len(current_group) > 1:
                    merged = self._merge_findings(current_group)
                    grouped.append(merged)
                else:
                    grouped.append(current_group[0])
                current_group = [curr]
        
        # Handle last group
        if len(current_group) > 1:
            merged = self._merge_findings(current_group)
            grouped.append(merged)
        else:
            grouped.append(current_group[0])
        
        return grouped
    
    def _merge_findings(self, findings: List[Finding]) -> Finding:
        """Merge multiple findings into one"""
        first = findings[0]
        last = findings[-1]
        
        # Merge metrics (average)
        merged_metrics = {}
        for key in first.metrics:
            values = [f.metrics.get(key, 0) for f in findings]
            merged_metrics[key] = sum(values) / len(values)
        
        # Collect all evidence
        all_evidence = []
        for f in findings:
            all_evidence.extend(f.evidence_artifacts)
        
        return Finding(
            id=str(uuid.uuid4()),
            type=first.type,
            severity = max(
                (f.severity for f in findings),
                key=lambda s: ['low', 'medium', 'high'].index(s.value)
            ),
            location={
                'start': first.location.get('start'),
                'end': last.location.get('end')
            },
            explanation=f"Merged segment with {len(findings)} anomalies. " + first.explanation,
            metrics=merged_metrics,
            evidence_artifacts=all_evidence[:6]  # Limit evidence
        )
