"""
Camera Tools - Fetch and display the latest camera image.

Provides a ROS 2 image subscription interface and a LangChain tool for
retrieving the most recent camera frame for display in the UI/chat.
"""

import base64
import io
import logging
import os
import sys
import threading
import time
from typing import Optional, Type, Any

import numpy as np
from PIL import Image, ImageDraw
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ranger_llm_ui.utils.logger import log_tool_call

logger = logging.getLogger(__name__)

# Try to import ROS 2, but allow running without it for testing
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from ros2_numpy.image import image_to_numpy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("ROS 2 (rclpy) not available. Running in simulation mode.")


# --------------------------------------------------------------------------- #
# Named cameras
#
# The Ranger + PiPER arm carry three physical cameras. The agent selects one by
# friendly name (the GetCameraImage 'camera' arg); each name maps to a capture
# source:
#   front  - Tier IV C2-176 fisheye, the forward/base camera. ROS topic.
#   wrist  - Intel RealSense D405 on the arm wrist. ROS topic (served by the
#            ranger-garden-assistant realsense node).
#   rear   - Intel RealSense D435i mounted BEHIND the arm. This one is NOT a ROS
#            node (only the wrist D405 streams over ROS); it is a free USB device
#            grabbed on demand by serial with pyrealsense2, then closed — exactly
#            like MobileManipulationCore's handover skill. See
#            _capture_realsense_rgb(). We deliberately do NOT wire it to ROS.
# Each source/topic/serial is overridable by env var so a relocated camera needs
# no code change.
# --------------------------------------------------------------------------- #
DEFAULT_CAMERA_TOPIC = "/camera/image_raw"
DEFAULT_WRIST_TOPIC = "/piper/wrist_camera/piper_d405/color/image_raw"
DEFAULT_REAR_SERIAL = "243722070013"  # the D435i behind the arm (MMC handover cam)

# Friendly aliases the LLM might say -> canonical camera name.
_CAMERA_ALIASES = {
    "base": "front", "main": "front", "default": "front", "fisheye": "front",
    "forward": "front", "nav": "front", "navigation": "front",
    "arm": "wrist", "hand": "wrist", "gripper": "wrist", "d405": "wrist",
    "back": "rear", "behind": "rear", "fixed": "rear", "handover": "rear",
    "d435": "rear", "d435i": "rear",
}


def _get_camera_config():
    """Get image-encoding configuration from environment variables."""
    return {
        "max_width": int(os.getenv("CAMERA_IMAGE_MAX_WIDTH", "320")),
        "max_height": int(os.getenv("CAMERA_IMAGE_MAX_HEIGHT", "240")),
        "quality": int(os.getenv("CAMERA_IMAGE_QUALITY", "75")),
        "format": os.getenv("CAMERA_IMAGE_FORMAT", "jpeg").lower(),
        "topic": os.getenv("CAMERA_TOPIC", DEFAULT_CAMERA_TOPIC),
    }


def _get_named_cameras() -> dict:
    """Named cameras the agent can select by name (each source env-overridable)."""
    return {
        "front": {"source": "ros",
                  "topic": os.getenv("CAMERA_TOPIC", DEFAULT_CAMERA_TOPIC)},
        "wrist": {"source": "ros",
                  "topic": os.getenv("CAMERA_WRIST_TOPIC", DEFAULT_WRIST_TOPIC)},
        "rear": {"source": "realsense",
                 "serial": os.getenv("CAMERA_REAR_SERIAL", DEFAULT_REAR_SERIAL)},
    }


def _default_camera_name() -> str:
    name = os.getenv("CAMERA_DEFAULT", "front").strip().lower()
    name = _CAMERA_ALIASES.get(name, name)
    return name if name in _get_named_cameras() else "front"


def _resolve_camera(camera: Optional[str], topic: Optional[str]) -> dict:
    """Resolve the requested camera into a capture spec.

    Precedence: an explicit ROS ``topic`` wins (back-compat); otherwise a friendly
    ``camera`` name is looked up in the named-camera registry; otherwise the
    default camera (``CAMERA_DEFAULT`` or 'front') is used. An unknown name that
    looks like a ROS topic ('/...') is used directly; any other unknown name falls
    back to the default camera. The returned spec always has 'name', 'source', a
    human-readable 'label', and either 'topic' (ros) or 'serial' (realsense).
    """
    cameras = _get_named_cameras()

    if topic:
        spec = {"name": topic, "source": "ros", "topic": topic}
    else:
        raw = (camera or _default_camera_name()).strip()
        key = _CAMERA_ALIASES.get(raw.lower(), raw.lower())
        if key in cameras:
            spec = {"name": key, **cameras[key]}
        elif raw.startswith("/"):
            spec = {"name": raw, "source": "ros", "topic": raw}
        else:
            key = _default_camera_name()
            spec = {"name": key, **cameras[key]}

    if spec["source"] == "realsense":
        spec["label"] = (f"RealSense D435i behind the arm, "
                         f"serial {spec.get('serial', '')}")
    else:
        spec["label"] = spec.get("topic", spec["name"])
    return spec


def _make_sim_image(label: str) -> np.ndarray:
    """Build a labeled placeholder frame (used when no live stream is available)."""
    width, height = 640, 480
    image = Image.new("RGB", (width, height), (24, 28, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle([16, 16, width - 16, height - 16], outline=(80, 90, 100), width=2)
    draw.text((32, 32), "Simulation Camera", fill=(210, 210, 210))
    draw.text((32, 56), label, fill=(170, 170, 170))
    draw.text((32, 80), "No live image stream available", fill=(140, 140, 140))
    return np.array(image)


# --------------------------------------------------------------------------- #
# On-demand RealSense capture (the fixed D435i mounted behind the arm)
#
# This camera is intentionally NOT wired to ROS. We open it by serial with
# pyrealsense2, drop a few warm-up frames so auto-exposure settles, grab ONE
# color frame, and close the device — no node, no topic, no continuous stream.
# Mirrors MobileManipulationCore's handover skill (skills/handover_skill.py:
# _grab_once / _reset_device). pyrealsense2 is an optional, hardware-only
# dependency: if it (or the device) is missing the tool degrades gracefully with
# a clear hint rather than crashing.
# --------------------------------------------------------------------------- #
# The D435i is single-owner; serialize opens so a UI snapshot can't race a manual
# refresh (or, worst case, an MMC handover) trying to open it at the same moment.
_realsense_lock = threading.Lock()


def _default_realsense_extra_site() -> str:
    """Find a site-packages holding pyrealsense2 if it isn't already importable.

    The UI commonly runs under a dedicated PYTHONUSERBASE that does NOT contain
    pyrealsense2, while the package is installed in the standard per-user site
    (~/.local/lib/pythonX.Y/site-packages) — the very path MMC's handover skill
    points at. Honor CAMERA_REALSENSE_EXTRA_SITE first; otherwise auto-probe that
    standard user site so the rear camera works without any env wiring.
    """
    explicit = os.getenv("CAMERA_REALSENSE_EXTRA_SITE", "").strip()
    if explicit:
        return explicit
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidate = os.path.expanduser(f"~/.local/lib/{pyver}/site-packages")
    if os.path.isdir(os.path.join(candidate, "pyrealsense2")):
        return candidate
    return ""


def _get_realsense_config() -> dict:
    """Capture params for the on-demand RealSense grab (env-overridable)."""
    return {
        "width": int(os.getenv("CAMERA_REAR_WIDTH", "640")),
        "height": int(os.getenv("CAMERA_REAR_HEIGHT", "480")),
        "fps": int(os.getenv("CAMERA_REAR_FPS", "30")),
        "warmup": int(os.getenv("CAMERA_REAR_WARMUP_FRAMES", "12")),
        "timeout_sec": float(os.getenv("CAMERA_REAR_TIMEOUT_SEC", "5.0")),
        # pyrealsense2 may live in a different site-packages than the UI's
        # PYTHONUSERBASE (Jetson runs the ROS env with PYTHONNOUSERSITE=1). Auto-
        # discovered; override with CAMERA_REALSENSE_EXTRA_SITE.
        "extra_site": _default_realsense_extra_site(),
    }


def _lazy_import(name: str, extra_site: str = ""):
    """Import a runtime-only dep, falling back to an extra site-packages path.

    numpy is already loaded by the time this runs, so appending the user site at
    runtime exposes pyrealsense2 WITHOUT swapping the loaded numpy. A plain import
    when ``extra_site`` is empty or the module is already importable.
    """
    try:
        return __import__(name)
    except ImportError:
        if not extra_site:
            raise
        if extra_site not in sys.path:
            sys.path.append(extra_site)
        return __import__(name)


def _reset_realsense_device(rs, serial: str) -> None:
    """Hardware-reset the D435i to recover its no-frames USB state (best effort)."""
    try:
        for d in rs.context().query_devices():
            if d.get_info(rs.camera_info.serial_number) == str(serial):
                d.hardware_reset()
                return
    except Exception:  # reset is a recovery attempt; never mask the real error
        pass


def _capture_realsense_rgb(
    serial: str,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup: int = 12,
    timeout_sec: float = 5.0,
    extra_site: str = "",
    retries: int = 2,
    reset_on_fail: bool = True,
    reset_wait_sec: float = 7.0,
) -> Optional[np.ndarray]:
    """Grab one color frame from a RealSense device by serial; return RGB uint8.

    Returns an (H,W,3) RGB uint8 array, or None if the device yields no color
    frame. Raises ImportError if pyrealsense2 is unavailable, or RuntimeError if
    every attempt times out / the device is busy. The D435i can drop into a
    no-frames state under rapid open/close cycling, so a failed grab is retried:
    a cheap re-open first, then a hardware reset (a known RealSense USB quirk).
    """
    rs = _lazy_import("pyrealsense2", extra_site)  # hardware-only dep; lazy

    def _grab_once() -> Optional[np.ndarray]:
        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(str(serial))
        config.enable_stream(rs.stream.color, int(width), int(height),
                             rs.format.bgr8, int(fps))
        pipeline.start(config)
        try:
            timeout_ms = max(1000, int(timeout_sec * 1000))
            frames = None
            for _ in range(max(1, int(warmup))):  # drop frames so AE settles
                frames = pipeline.wait_for_frames(timeout_ms)
            color_frame = frames.get_color_frame() if frames else None
            if not color_frame:
                return None
            color_bgr = np.asanyarray(color_frame.get_data())  # H,W,3 BGR uint8
            return np.ascontiguousarray(color_bgr[..., ::-1])  # BGR -> RGB
        finally:
            pipeline.stop()

    with _realsense_lock:
        last_err: Optional[Exception] = None
        for attempt in range(int(retries) + 1):
            try:
                return _grab_once()
            except RuntimeError as e:  # frame timeout, device busy, ...
                last_err = e
                if attempt >= int(retries):
                    break
                # Escalate: cheap re-open first, hardware reset on later retries.
                if reset_on_fail and serial and attempt >= 1:
                    _reset_realsense_device(rs, serial)
                    time.sleep(float(reset_wait_sec))  # USB re-enumeration
                else:
                    time.sleep(1.5)
        raise last_err if last_err else RuntimeError("RealSense capture failed")


# --------------------------------------------------------------------------- #
# Optional image description
#
# GetCameraImage uses return_direct=True, so the captured frame is handed
# straight to the UI and the agent loop ends — the agent's own model never sees
# the picture. To answer "describe what you see", the tool makes ONE short vision
# call on the (already downscaled) frame using the agent's configured LLM, and
# uses that description as the caption above the inline image. Disable with
# CAMERA_DESCRIBE=false; customize the instruction with CAMERA_DESCRIBE_PROMPT.
# Best-effort: if no LLM is registered or the provider/model can't do vision, the
# tool falls back to a plain caption (and still shows the image).
# --------------------------------------------------------------------------- #
_describe_llm: Optional[Any] = None

DEFAULT_DESCRIBE_PROMPT = (
    "You are the robot looking through your {camera} camera. In 1-2 short "
    "sentences, describe what you see from your own first-person point of view "
    "(start with 'I can see'). Be concrete and factual; do not mention pixels, "
    "tokens, files, resolution, or that this is an image."
)


def set_describe_llm(llm: Any) -> None:
    """Register the (vision-capable) LLM the camera tool uses to describe frames."""
    global _describe_llm
    _describe_llm = llm


def _describe_enabled() -> bool:
    return os.getenv("CAMERA_DESCRIBE", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _describe_image(data_url: str, camera_name: str) -> Optional[str]:
    """Return a short natural-language description of the frame, or None.

    None means "no description available" — the caller falls back to a plain
    caption. Never raises: a model/provider without vision just yields None.
    """
    if _describe_llm is None or not _describe_enabled():
        return None
    prompt = os.getenv("CAMERA_DESCRIBE_PROMPT", DEFAULT_DESCRIBE_PROMPT)
    try:
        prompt = prompt.format(camera=camera_name)
    except Exception:  # a custom prompt without the {camera} field is fine
        pass
    try:
        from langchain_core.messages import HumanMessage

        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])
        resp = _describe_llm.invoke([msg])
        text = getattr(resp, "content", resp)
        if isinstance(text, list):  # some providers return a list of content blocks
            text = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in text
            )
        text = (text or "").strip()
        return text or None
    except Exception as e:
        logger.warning(
            "Camera image description failed (%s); using a plain caption.", e
        )
        return None


class ROSCameraInterface:
    """
    Singleton interface for camera image retrieval.

    Manages a subscription to a ROS 2 Image topic and exposes the latest frame.
    """

    _instance: Optional["ROSCameraInterface"] = None
    _node: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, node: Optional[Any] = None, topic: Optional[str] = None):
        """Initialize the camera interface with a node and topic."""
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()

        logger.info(f"Camera interface initialize called: node={node is not None}, topic={topic}, already_initialized={hasattr(self, '_initialized') and self._initialized}")

        if self._initialized:
            logger.info(f"Camera interface already initialized: current_node={self._node is not None}, new_node={node is not None}, simulation_mode={self._simulation_mode}")
            if node is not None and node is not self._node:
                logger.info("Updating camera interface with new ROS node")
                self._node = node
                # Update simulation mode based on new node
                self._simulation_mode = not ROS_AVAILABLE or node is None
                # Set up subscription with the new node
                if not self._simulation_mode:
                    logger.info(f"Setting up camera subscription on {self._topic}")
                    self._setup_subscription()
                    logger.info(f"ROS camera interface re-initialized with new node on topic: {self._topic}")
            if topic:
                self.set_topic(topic)
            return

        self._node = node
        self._image_sub = None
        self._latest_image_msg: Optional[Any] = None
        self._simulation_mode = not ROS_AVAILABLE or node is None
        self._topic = topic or os.getenv("CAMERA_TOPIC", DEFAULT_CAMERA_TOPIC)
        self._sim_image: Optional[np.ndarray] = None
        self._sim_image_topic = None

        logger.info(f"First-time camera interface initialization: simulation_mode={self._simulation_mode}, topic={self._topic}")

        if not self._simulation_mode and self._node is not None:
            self._setup_subscription()
            logger.info(f"ROS camera interface initialized on topic: {self._topic}")
        else:
            logger.info("ROS camera interface running in simulation mode")

        self._initialized = True

    def _setup_subscription(self):
        if self._simulation_mode or self._node is None:
            return

        if self._image_sub is not None:
            try:
                self._node.destroy_subscription(self._image_sub)
            except Exception as e:
                logger.debug(f"Failed to destroy old camera subscription: {e}")

        self._image_sub = self._node.create_subscription(
            RosImage,
            self._topic,
            self._image_callback,
            10,
        )

    def _image_callback(self, msg):
        """Callback for camera image messages."""
        with self._lock:
            self._latest_image_msg = msg

    @property
    def simulation_mode(self) -> bool:
        return self._simulation_mode

    @property
    def topic(self) -> str:
        return self._topic

    def set_topic(self, topic: str):
        """Update the camera topic subscription."""
        if not topic:
            return
        self._topic = topic
        if not self._simulation_mode and self._node is not None:
            self._setup_subscription()
        else:
            # Regenerate simulated image with new topic label
            self._sim_image = None
            self._sim_image_topic = None

    def get_latest_image(self) -> Optional[np.ndarray]:
        """Return the latest camera image as a numpy array."""
        if self._simulation_mode:
            return self._get_simulated_image()

        with self._lock:
            msg = self._latest_image_msg

        if msg is None:
            return None

        image = self._convert_ros_image(msg)
        if image is None:
            return None

        return self._normalize_image(image, getattr(msg, "encoding", ""))

    def _convert_ros_image(self, msg: Any) -> Optional[np.ndarray]:
        """Convert a ROS image message into a numpy array."""
        encoding = (getattr(msg, "encoding", "") or "").lower()

        try:
            return image_to_numpy(msg)
        except Exception as e:
            if encoding in {
                "yuv422",
                "yuyv",
                "yuyv422",
                "yuv422_yuy2",
                "yuy2",
                "uyvy",
                "uyvy422",
            }:
                image = self._convert_yuv422_to_rgb(msg, encoding)
                if image is not None:
                    logger.info(
                        f"Converted ROS image with fallback decoder for encoding '{encoding}'"
                    )
                    return image
            logger.error(f"Failed to convert ROS image: {e}")
            return None

    def _convert_yuv422_to_rgb(self, msg: Any, encoding: str) -> Optional[np.ndarray]:
        """Decode YUV422-family encodings into RGB."""
        width = int(getattr(msg, "width", 0) or 0)
        height = int(getattr(msg, "height", 0) or 0)
        if width <= 0 or height <= 0:
            logger.error(
                f"Invalid YUV422 image dimensions: width={width}, height={height}"
            )
            return None

        if width % 2 != 0:
            logger.error(f"YUV422 requires an even width, got {width}")
            return None

        bytes_per_row = width * 2
        step = int(getattr(msg, "step", 0) or bytes_per_row)
        if step < bytes_per_row:
            logger.error(
                f"Invalid YUV422 row step: step={step}, expected at least {bytes_per_row}"
            )
            return None

        data = np.frombuffer(msg.data, dtype=np.uint8)
        expected_size = step * height
        if data.size < expected_size:
            logger.error(
                f"YUV422 data too small: got {data.size} bytes, expected {expected_size}"
            )
            return None

        rows = data[:expected_size].reshape(height, step)
        pairs = rows[:, :bytes_per_row].reshape(height, width // 2, 4)

        if encoding in {"yuyv", "yuyv422", "yuv422_yuy2", "yuy2"}:
            layout = "yuyv"
        elif encoding in {"uyvy", "uyvy422"}:
            layout = "uyvy"
        else:
            # "yuv422" is ambiguous in practice; infer layout from luma variance.
            even_var = float(np.var(pairs[..., [0, 2]].astype(np.float32)))
            odd_var = float(np.var(pairs[..., [1, 3]].astype(np.float32)))
            layout = "uyvy" if odd_var > even_var else "yuyv"

        if layout == "yuyv":
            y0 = pairs[..., 0].astype(np.float32)
            u = pairs[..., 1].astype(np.float32)
            y1 = pairs[..., 2].astype(np.float32)
            v = pairs[..., 3].astype(np.float32)
        else:
            u = pairs[..., 0].astype(np.float32)
            y0 = pairs[..., 1].astype(np.float32)
            v = pairs[..., 2].astype(np.float32)
            y1 = pairs[..., 3].astype(np.float32)

        y = np.empty((height, width), dtype=np.float32)
        y[:, 0::2] = y0
        y[:, 1::2] = y1
        u = np.repeat(u, 2, axis=1) - 128.0
        v = np.repeat(v, 2, axis=1) - 128.0

        r = y + (1.402 * v)
        g = y - (0.344136 * u) - (0.714136 * v)
        b = y + (1.772 * u)

        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    def _get_simulated_image(self) -> np.ndarray:
        """Generate or return a cached simulated camera image."""
        if self._sim_image is not None and self._sim_image_topic == self._topic:
            return self._sim_image

        self._sim_image = _make_sim_image(f"Topic: {self._topic}")
        self._sim_image_topic = self._topic
        return self._sim_image

    def _normalize_image(self, image: np.ndarray, encoding: str) -> np.ndarray:
        """Normalize image to RGB uint8 for UI display."""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        encoding = (encoding or "").lower()
        if image.ndim == 3 and image.shape[2] >= 3:
            if encoding.startswith("bgr"):
                image = image[..., [2, 1, 0]]
            else:
                image = image[..., :3]

        if image.dtype != np.uint8:
            image = self._scale_to_uint8(image)

        return image

    def _scale_to_uint8(self, image: np.ndarray) -> np.ndarray:
        if np.issubdtype(image.dtype, np.floating):
            image = np.nan_to_num(image)
            image = np.clip(image, 0.0, 1.0)
            return (image * 255.0).astype(np.uint8)

        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            if info.max == 0:
                return np.zeros_like(image, dtype=np.uint8)
            scaled = (image.astype(np.float32) / float(info.max)) * 255.0
            return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

        return image.astype(np.uint8)


_camera_interface: Optional[ROSCameraInterface] = None


def get_camera_interface() -> ROSCameraInterface:
    """Get or create the ROS camera interface singleton."""
    global _camera_interface
    if _camera_interface is None:
        _camera_interface = ROSCameraInterface()
    # Ensure interface is initialized (with no node if not already initialized)
    if not _camera_interface._initialized:
        _camera_interface.initialize()
    return _camera_interface


def initialize_camera_interface(node: Optional[Any] = None, topic: Optional[str] = None):
    """Initialize the camera interface with a node."""
    interface = get_camera_interface()
    interface.initialize(node, topic=topic)
    return interface


class CameraImageInput(BaseModel):
    """Input schema for camera image capture.

    All parameters are optional. If not provided, defaults from environment
    variables or config file will be used (320x240 JPEG quality=75).
    """

    camera: Optional[str] = Field(
        default=None,
        description=(
            "Which camera to view, by name: 'front' (the forward/base fisheye, "
            "the default), 'wrist' (the arm/wrist camera — use this for 'wrist "
            "cam' / 'arm camera' requests), or 'rear' (the fixed camera mounted "
            "behind the arm). If omitted, the default ('front') camera is used."
        ),
    )
    topic: Optional[str] = Field(
        default=None,
        description=(
            "Advanced: an explicit ROS 2 camera topic (sensor_msgs/Image) to read "
            "from, overriding 'camera'. Leave unset and use 'camera' for the "
            "named views."
        ),
    )
    max_width: Optional[int] = Field(
        default=None,
        ge=64,
        le=1920,
        description="Max width for the returned image in pixels. If not specified, uses CAMERA_IMAGE_MAX_WIDTH env var or 320.",
    )
    max_height: Optional[int] = Field(
        default=None,
        ge=64,
        le=1080,
        description="Max height for the returned image in pixels. If not specified, uses CAMERA_IMAGE_MAX_HEIGHT env var or 240.",
    )
    quality: Optional[int] = Field(
        default=None,
        ge=10,
        le=100,
        description="JPEG compression quality (10-100). Higher = better quality but more tokens. If not specified, uses CAMERA_IMAGE_QUALITY env var or 75.",
    )
    format: Optional[str] = Field(
        default=None,
        description="Image format: 'jpeg' (smaller, lossy) or 'png' (larger, lossless). If not specified, uses CAMERA_IMAGE_FORMAT env var or 'jpeg'.",
    )


class GetCameraImageTool(BaseTool):
    """Tool to fetch the latest camera image."""

    name: str = "GetCameraImage"
    description: str = (
        "Get the latest camera image from one of the robot's cameras and show it "
        "in the chat. Pick the view with 'camera': 'front' (forward/base fisheye, "
        "default), 'wrist' (the arm/wrist camera — use for 'show the wrist cam'), "
        "or 'rear' (the fixed camera behind the arm). Use this whenever the "
        "operator asks to see a camera. All parameters are optional. Returns a "
        "reduced-resolution image embedded in markdown."
    )
    args_schema: Type[BaseModel] = CameraImageInput
    return_direct: bool = True

    def _run(
        self,
        camera: Optional[str] = None,
        topic: Optional[str] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        quality: Optional[int] = None,
        format: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Fetch the latest image from the selected camera and return it as Markdown."""
        start_time = time.time()

        # Get defaults from config if not provided
        config = _get_camera_config()
        max_width = max_width if max_width is not None else config["max_width"]
        max_height = max_height if max_height is not None else config["max_height"]
        quality = quality if quality is not None else config["quality"]
        format = format if format is not None else config["format"]

        spec = _resolve_camera(camera, topic)
        label = spec["label"]
        interface = get_camera_interface()
        log_params = {
            "camera": camera, "topic": topic, "resolved": spec["name"],
            "source": spec["source"], "max_width": max_width,
            "max_height": max_height, "quality": quality, "format": format,
        }

        # ---- acquire one frame from the resolved source --------------------
        try:
            if interface.simulation_mode:
                # No live streams in --simple mode: labeled placeholder per camera.
                image = _make_sim_image(label)
            elif spec["source"] == "realsense":
                # The fixed D435i behind the arm: grabbed on demand, not via ROS.
                image = _capture_realsense_rgb(
                    serial=spec.get("serial", ""), **_get_realsense_config()
                )
            else:  # ROS topic (front / wrist / explicit topic)
                changed = interface.topic != spec["topic"]
                interface.set_topic(spec["topic"])
                image = interface.get_latest_image()
                if image is None and changed:
                    # Subscription was just (re)created; give the first frame a
                    # moment to arrive so a camera switch works on the first try.
                    deadline = time.time() + float(
                        os.getenv("CAMERA_SWITCH_WAIT_SEC", "3.0")
                    )
                    while image is None and time.time() < deadline:
                        time.sleep(0.1)
                        image = interface.get_latest_image()
        except Exception as e:
            result = f"Failed to capture the '{spec['name']}' camera ({label}): {e}"
            if spec["source"] == "realsense":
                result += (
                    " — the rear D435i is read on demand via pyrealsense2 (it is "
                    "not a ROS stream). Ensure pyrealsense2 is installed, the "
                    f"device (serial {spec.get('serial', '')}) is connected and "
                    "not already in use, or set CAMERA_REAR_SERIAL / "
                    "CAMERA_REALSENSE_EXTRA_SITE."
                )
            log_tool_call(
                tool_name=self.name, parameters=log_params, result=result,
                success=False, error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return result

        if image is None:
            if spec["source"] == "realsense":
                result = (
                    f"No image from the '{spec['name']}' camera ({label}); the "
                    "D435i returned no color frame."
                )
            else:
                result = f"No camera image available yet on {spec.get('topic', label)}."
            log_tool_call(
                tool_name=self.name, parameters=log_params, result=result,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return result

        try:
            pil_image = Image.fromarray(image)
            pil_image.thumbnail((max_width, max_height))
            buffer = io.BytesIO()

            # Normalize format string
            img_format = format.lower().strip()
            if img_format == "jpeg" or img_format == "jpg":
                # JPEG format with quality compression
                pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
                mime_type = "image/jpeg"
            else:
                # PNG format (lossless)
                pil_image.save(buffer, format="PNG", optimize=True)
                mime_type = "image/png"

            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            encoded_size_kb = len(encoded) / 1024
            estimated_tokens = int(len(encoded) / 4)  # Rough estimate: 4 bytes ≈ 1 token

            data_url = f"data:{mime_type};base64,{encoded}"

            # Caption: a natural-language description of what the camera sees
            # (the agent never sees the frame itself — return_direct ends the
            # turn). Falls back to a plain caption when description is off/
            # unavailable. The technical details (topic, size, tokens) stay in
            # the logs, out of the operator's chat.
            description = _describe_image(data_url, spec["name"])
            caption = description or f"Here is my {spec['name']} camera view."
            result = f"{caption}\n\n![Camera Image]({data_url})"

            logger.info(
                f"Camera image encoded: camera={spec['name']} ({label}) "
                f"{pil_image.width}x{pil_image.height} {img_format.upper()}, "
                f"size={encoded_size_kb:.1f}KB, est_tokens={estimated_tokens:,}, "
                f"described={'yes' if description else 'no'}"
            )

            log_tool_call(
                tool_name=self.name,
                parameters=log_params,
                result=f"Camera image captured from {spec['name']} ({pil_image.width}x{pil_image.height}, {img_format.upper()}, ~{estimated_tokens:,} tokens)",
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return result
        except Exception as e:
            result = f"Failed to render camera image: {e}"
            log_tool_call(
                tool_name=self.name,
                parameters=log_params,
                result=result,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return result


def get_camera_tools() -> list[BaseTool]:
    """Get all camera-related tools."""
    return [GetCameraImageTool()]
