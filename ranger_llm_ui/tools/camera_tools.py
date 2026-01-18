"""
Camera Tools - Fetch and display the latest camera image.

Provides a ROS 2 image subscription interface and a LangChain tool for
retrieving the most recent camera frame for display in the UI/chat.
"""

import base64
import io
import logging
import os
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


DEFAULT_CAMERA_TOPIC = "/camera/image_raw"


def _get_camera_config():
    """Get camera configuration from environment variables."""
    return {
        "max_width": int(os.getenv("CAMERA_IMAGE_MAX_WIDTH", "320")),
        "max_height": int(os.getenv("CAMERA_IMAGE_MAX_HEIGHT", "240")),
        "quality": int(os.getenv("CAMERA_IMAGE_QUALITY", "75")),
        "format": os.getenv("CAMERA_IMAGE_FORMAT", "jpeg").lower(),
        "topic": os.getenv("CAMERA_TOPIC", DEFAULT_CAMERA_TOPIC),
    }


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

        try:
            image = image_to_numpy(msg)
        except Exception as e:
            logger.error(f"Failed to convert ROS image: {e}")
            return None

        return self._normalize_image(image, getattr(msg, "encoding", ""))

    def _get_simulated_image(self) -> np.ndarray:
        """Generate or return a cached simulated camera image."""
        if self._sim_image is not None and self._sim_image_topic == self._topic:
            return self._sim_image

        width, height = 640, 480
        image = Image.new("RGB", (width, height), (24, 28, 32))
        draw = ImageDraw.Draw(image)
        draw.rectangle([16, 16, width - 16, height - 16], outline=(80, 90, 100), width=2)
        draw.text((32, 32), "Simulation Camera", fill=(210, 210, 210))
        draw.text((32, 56), f"Topic: {self._topic}", fill=(170, 170, 170))
        draw.text((32, 80), "No ROS 2 image stream available", fill=(140, 140, 140))

        self._sim_image = np.array(image)
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

    topic: Optional[str] = Field(
        default=None,
        description="Optional ROS 2 camera topic to read from (sensor_msgs/Image). If not specified, uses default topic.",
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
        "Get the latest camera image from the robot's camera topic. "
        "Use this when you need a current camera snapshot. "
        "All parameters are optional - defaults will be used if not specified. "
        "Returns a base64-encoded image embedded in markdown."
    )
    args_schema: Type[BaseModel] = CameraImageInput
    return_direct: bool = True

    def _run(
        self,
        topic: Optional[str] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        quality: Optional[int] = None,
        format: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Fetch the latest camera image and return it as Markdown."""
        start_time = time.time()

        # Get defaults from config if not provided
        config = _get_camera_config()
        max_width = max_width if max_width is not None else config["max_width"]
        max_height = max_height if max_height is not None else config["max_height"]
        quality = quality if quality is not None else config["quality"]
        format = format if format is not None else config["format"]

        interface = get_camera_interface()
        if topic:
            interface.set_topic(topic)

        image = interface.get_latest_image()

        if image is None:
            result = f"No camera image available yet on {interface.topic}."
            log_tool_call(
                tool_name=self.name,
                parameters={"topic": topic, "max_width": max_width, "max_height": max_height, "quality": quality, "format": format},
                result=result,
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
            result = (
                f"Camera image from {interface.topic} ({pil_image.width}x{pil_image.height}, "
                f"{img_format.upper()}, ~{estimated_tokens:,} tokens).\n\n"
                f"![Camera Image]({data_url})"
            )

            logger.info(
                f"Camera image encoded: {pil_image.width}x{pil_image.height} {img_format.upper()}, "
                f"size={encoded_size_kb:.1f}KB, est_tokens={estimated_tokens:,}"
            )

            log_tool_call(
                tool_name=self.name,
                parameters={"topic": topic, "max_width": max_width, "max_height": max_height, "quality": quality, "format": format},
                result=f"Camera image captured ({pil_image.width}x{pil_image.height}, {img_format.upper()}, ~{estimated_tokens:,} tokens)",
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return result
        except Exception as e:
            result = f"Failed to render camera image: {e}"
            log_tool_call(
                tool_name=self.name,
                parameters={"topic": topic, "max_width": max_width, "max_height": max_height, "quality": quality, "format": format},
                result=result,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return result


def get_camera_tools() -> list[BaseTool]:
    """Get all camera-related tools."""
    return [GetCameraImageTool()]
