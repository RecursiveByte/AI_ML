import { useState, useRef, useEffect } from 'react';
import { useMediaPipe } from './useMediaPipe';
import { PREDICTION_INTERVAL, VIDEO_CONFIG } from '../config/constants';

export const useWebcam = (apiUrl, backendStatus) => {
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [currentLetter, setCurrentLetter] = useState("-");
  const [confidence, setConfidence] = useState(0);
  const [handCount, setHandCount] = useState(0);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const frameIntervalRef = useRef(null);

  const { initMediaPipe, stopMediaPipe } = useMediaPipe(videoRef, canvasRef);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: VIDEO_CONFIG
      });

      videoRef.current.srcObject = stream;
      
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play();
        setIsWebcamActive(true);
        initMediaPipe();
        startFrameCapture();
      };
    } catch (err) {
      console.error("Webcam error:", err);
      alert("Webcam permission denied or not available");
    }
  };

  const stopWebcam = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }

    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    stopMediaPipe();
    setIsWebcamActive(false);
    
    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const startFrameCapture = () => {
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    
    frameIntervalRef.current = setInterval(() => {
      const video = videoRef.current;
      if (!video || !video.videoWidth) return;

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Mirror the image for backend
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0);

      sendFrameToBackend(canvas.toDataURL("image/jpeg", 0.8));
    }, PREDICTION_INTERVAL);
  };

  const sendFrameToBackend = async (image) => {
    try {
      const res = await fetch(`${apiUrl}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image })
      });
      const d = await res.json();

      const predictedLetter = d.predicted_letter || d.current_letter || "-";
      setCurrentLetter(predictedLetter);
      setConfidence(d.confidence || 0);
      setHandCount(d.hand_count || 0);
      
    } catch (err) {
      console.error("Backend error:", err);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  return {
    isWebcamActive,
    videoRef,
    canvasRef,
    currentLetter,
    confidence,
    handCount,
    startWebcam,
    stopWebcam
  };
};