import { useRef } from 'react';
import { drawHandLandmarks } from '../utils/handDrawing';
import { MEDIAPIPE_CONFIG } from '../config/constants';

export const useMediaPipe = (videoRef, canvasRef) => {
  const handsRef = useRef(null);
  const cameraRef = useRef(null);
  const animationRef = useRef(null);

  const initMediaPipe = async () => {
    try {
      // Import modules
      const handsModule = await import("@mediapipe/hands");
      const cameraModule = await import("@mediapipe/camera_utils");

      // Handle both default and named exports
      const Hands = handsModule.Hands || handsModule.default?.Hands || handsModule.default;
      const Camera = cameraModule.Camera || cameraModule.default?.Camera || cameraModule.default;

      if (!Hands || !Camera) {
        throw new Error("Failed to load MediaPipe modules");
      }

      console.log("MediaPipe modules loaded successfully");

      const hands = new Hands({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });

      hands.setOptions(MEDIAPIPE_CONFIG);

      hands.onResults((results) => {
        if (canvasRef.current) {
          drawHandLandmarks(canvasRef.current, results, videoRef.current);
        }
      });

      handsRef.current = hands;

      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          if (handsRef.current && videoRef.current) {
            await handsRef.current.send({ image: videoRef.current });
          }
        },
        width: 1280,
        height: 720,
      });

      cameraRef.current = camera;
      camera.start();
      
      console.log("MediaPipe initialized successfully");
      
    } catch (err) {
      console.error("MediaPipe initialization error:", err);
      alert("Failed to initialize hand detection. Please refresh the page.");
    }
  };

  const stopMediaPipe = () => {
    if (cameraRef.current) {
      cameraRef.current.stop();
      cameraRef.current = null;
    }

    if (handsRef.current) {
      handsRef.current.close();
      handsRef.current = null;
    }

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  };

  return {
    initMediaPipe,
    stopMediaPipe
  };
};