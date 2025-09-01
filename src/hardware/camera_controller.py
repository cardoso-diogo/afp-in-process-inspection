# src/hardware/camera_controller.py

import sys
import queue
from vimba import Vimba, PixelFormat, Frame, VimbaTimeout, FrameStatus
from threading import Event


class FrameHandler:
    def __init__(self, frame_queue):
        self.frame_queue = frame_queue
        # We use a threading Event to signal when streaming should stop for this session.
        self.shutdown_event = Event()

    def __call__(self, cam, frame: Frame):
        # This is the Vimba streaming callback.
        # If the shutdown signal has been given, ignore all new frames.
        if self.shutdown_event.is_set():
            cam.queue_frame(frame)
            return

        # If the frame was acquired successfully, put its data in the queue.
        if frame.get_status() == FrameStatus.Complete:
            try:
                # Use put_nowait because this callback should never block.
                # If the main loop can't keep up, we drop the frame.
                self.frame_queue.put_nowait(frame.as_opencv_image().copy())
            except queue.Full:
                # This is not an error, just an indication that the consumer is slow.
                # We silently drop the frame and continue.
                pass

        # Re-queue the Vimba frame buffer so it can be used for the next image.
        cam.queue_frame(frame)


class CameraController:
    """
    A controller class for the Alvium camera using a robust, session-based FrameHandler
    to allow for multiple start/stop cycles within a single application run.
    """

    def __init__(self):
        self.vimba = None
        self.cam = None
        self.is_streaming = False
        self.frame_handler = None
        # Store the queue so we can re-create the handler for each new session
        self.frame_queue = None

    def connect(self, frame_queue):
        """
        Initializes the Vimba SDK, connects to the first available camera, and
        stores the frame queue for later use.
        """
        print("Camera Controller: Attempting to connect...")
        self.frame_queue = frame_queue
        try:
            self.vimba = Vimba.get_instance()
            self.vimba.__enter__()

            cams = self.vimba.get_all_cameras()
            if not cams:
                raise ConnectionError("No cameras found. Please check connection.")

            self.cam = cams[0]
            self.cam.__enter__()

            self._configure_camera()
            print("Camera Controller: Connected and configured successfully.")
            return True
        except Exception as e:
            print(f"Camera Controller: Connection failed - {e}", file=sys.stderr)
            self.disconnect()  # Ensure a clean exit on failure
            return False

    def _configure_camera(self):
        """Sets essential camera parameters like pixel format and packet size."""
        try:
            self.cam.set_pixel_format(PixelFormat.Bgr8)
            print("Camera Controller: Pixel format set to Bgr8.")
        except Exception:
            self.cam.set_pixel_format(PixelFormat.Mono8)
            print("Camera Controller: Pixel format set to Mono8.")

        # Optimize packet size for GigE cameras to prevent dropped frames
        try:
            self.cam.GVSPAdjustPacketSize.run()
            while not self.cam.GVSPAdjustPacketSize.is_done():
                pass
            print("Camera Controller: GVSP Packet Size adjusted.")
        except (AttributeError, VimbaTimeout):
            # This is expected if it's not a GigE camera
            pass

    def start_streaming(self):
        """Starts a new streaming session with a fresh FrameHandler."""
        if self.cam and not self.is_streaming:
            if self.frame_queue is None:
                print("Camera Controller: Error - Frame queue not provided. Cannot start streaming.", file=sys.stderr)
                return

            try:
                self.frame_handler = FrameHandler(self.frame_queue)

                # buffer_count allocates internal buffers for Vimba to use.
                self.cam.start_streaming(handler=self.frame_handler, buffer_count=10)
                self.is_streaming = True
                print("Camera Controller: Asynchronous streaming started.")
            except Exception as e:
                print(f"Camera Controller: Could not start streaming - {e}", file=sys.stderr)

    def stop_streaming(self):
        """Stops the current streaming session and signals the handler to shut down."""
        if self.cam and self.is_streaming:
            # Signal the current frame handler to stop processing new frames
            if self.frame_handler:
                self.frame_handler.shutdown_event.set()

            try:
                # This call stops the camera's internal acquisition threads
                self.cam.stop_streaming()
                self.is_streaming = False
                print("Camera Controller: Streaming stopped.")
            except Exception as e:
                print(f"Camera Controller: Could not stop streaming - {e}", file=sys.stderr)

    def disconnect(self):
        """Stops streaming and properly closes all Vimba resources."""
        print("Camera Controller: Disconnecting...")
        if self.cam:
            self.stop_streaming()
            try:
                self.cam.__exit__(None, None, None)
            except Exception:
                pass
            self.cam = None

        if self.vimba:
            try:
                self.vimba.__exit__(None, None, None)
            except Exception:
                pass
            self.vimba = None
        print("Camera Controller: Disconnected.")