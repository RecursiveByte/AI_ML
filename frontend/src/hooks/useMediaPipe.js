import { useRef } from 'react';
import { drawHandLandmarks } from '../utils/handDrawing';
import { MEDIAPIPE_CONFIG } from '../config/constants';

export const useMediaPipe = (videoRef, canvasRef) => {
  const handsRef = useRef(null);
  const cameraRef = useRef(null);
  const animationRef = useRef(null);

  const initMediaPipe = async () => {
    try {
      const { Hands } = await import("@mediapipe/hands");
      const { Camera } = await import("@mediapipe/camera_utils");

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
      
    } catch (err) {
      console.error("MediaPipe initialization error:", err);
    }
  };

  const stopMediaPipe = () => {
    if (cameraRef.current) {
      cameraRef.current.stop();
      cameraRef.current = null;
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