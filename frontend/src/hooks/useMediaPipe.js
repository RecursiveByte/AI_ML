import { useRef } from 'react';
import { drawHandLandmarks } from '../utils/handDrawing';
import { MEDIAPIPE_CONFIG } from '../config/constants';

export const useMediaPipe = (videoRef, canvasRef) => {
  const handsRef = useRef(null);
  const cameraRef = useRef(null);

  const loadScript = (src) => {
    return new Promise((resolve, reject) => {
      // Check if already loaded
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve();
        return;
      }

      const script = document.createElement('script');
      script.src = src;
      script.crossOrigin = 'anonymous';
      script.onload = resolve;
      script.onerror = reject;
      document.body.appendChild(script);
    });
  };

  const initMediaPipe = async () => {
    try {
      console.log("Loading MediaPipe from CDN...");

      // Load MediaPipe scripts from CDN
      await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js');
      await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js');
      await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js');
      await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js');

      console.log("MediaPipe scripts loaded");

      // Wait a bit for scripts to initialize
      await new Promise(resolve => setTimeout(resolve, 100));

      // Access from window object
      const { Hands } = window;
      const { Camera } = window;

      if (!Hands || !Camera) {
        throw new Error("MediaPipe not available on window object");
      }

      const hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
      });

      hands.setOptions(MEDIAPIPE_CONFIG);

      hands.onResults((results) => {
        if (canvasRef.current && videoRef.current) {
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

      console.log("MediaPipe initialized successfully!");

    } catch (err) {
      console.error("MediaPipe initialization error:", err);
      alert(`Hand detection failed to load: ${err.message}. Please refresh the page.`);
    }
  };

  const stopMediaPipe = () => {
    if (cameraRef.current) {
      cameraRef.current.stop();
      cameraRef.current = null;
    }

    if (handsRef.current) {
      handsRef.current.close?.();
      handsRef.current = null;
    }
  };

  return {
    initMediaPipe,
    stopMediaPipe
  };
};