import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// API Configuration
const API_URL = 'https://ai-ml-bmmn.onrender.com';
const FRAME_INTERVAL = 1000; // Send frame every 1 second

function App() {
  // State management
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [currentLetter, setCurrentLetter] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [handCount, setHandCount] = useState(0);
  const [currentWord, setCurrentWord] = useState('[empty]');
  const [countdown, setCountdown] = useState(5.0);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [statusMessage, setStatusMessage] = useState('Checking backend...');

  // Refs for webcam and canvas
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const frameIntervalRef = useRef(null);

  // Check backend connection on mount
  useEffect(() => {
    checkBackendConnection();
  }, []);

  // Check if backend is running
  const checkBackendConnection = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`);
      if (response.data.status === 'healthy') {
        setBackendStatus('connected');
        setStatusMessage('Connected to backend - Ready to start');
      } else {
        setBackendStatus('error');
        setStatusMessage('Backend not ready - Models not loaded');
      }
    } catch (error) {
      setBackendStatus('error');
      setStatusMessage('Cannot connect to backend - Make sure app.py is running');
      console.error('Backend connection error:', error);
    }
  };

  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsWebcamActive(true);
        setStatusMessage('Webcam active - Show hand gestures');
        
        // Start sending frames to backend
        startFrameCapture();
      }
    } catch (error) {
      console.error('Webcam error:', error);
      setStatusMessage('Failed to access webcam');
      alert('Could not access webcam. Please check permissions.');
    }
  };

  // Stop webcam
  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
    }
    
    setIsWebcamActive(false);
    setStatusMessage('Webcam stopped');
  };

  // Capture and send frames to backend
  const startFrameCapture = () => {
    frameIntervalRef.current = setInterval(() => {
      if (videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        
        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw video frame to canvas (flipped for mirror effect)
        context.translate(canvas.width, 0);
        context.scale(-1, 1);
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        context.setTransform(1, 0, 0, 1, 0, 0); // Reset transform
        
        // Convert canvas to base64
        const base64Image = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to backend
        sendFrameToBackend(base64Image);
      }
    }, FRAME_INTERVAL);
  };

  // Send frame to backend API
  const sendFrameToBackend = async (base64Image) => {
    try {
      const response = await axios.post(`${API_URL}/api/predict`, {
        image: base64Image
      });
      
      const data = response.data;
      
      // Update UI with response
      setCurrentLetter(data.current_letter || '-');
      setConfidence(data.confidence);
      setHandCount(data.hand_count);
      setCurrentWord(data.current_word || '[empty]');
      setCountdown(data.time_until_next_add);
      
      // Show notification if letter was added
      if (data.letter_added) {
        console.log(`[Added] '${data.letter_added}' → Word: '${data.current_word}'`);
      }
    } catch (error) {
      console.error('Prediction error:', error);
      if (backendStatus !== 'error') {
        setBackendStatus('error');
        setStatusMessage('Connection error - Check if backend is running');
      }
    }
  };

  // Word control functions
  const addSpace = async () => {
    try {
      const response = await axios.post(`${API_URL}/api/word/space`);
      setCurrentWord(response.data.current_word || '[empty]');
    } catch (error) {
      console.error('Error adding space:', error);
    }
  };

  const deleteChar = async () => {
    try {
      const response = await axios.post(`${API_URL}/api/word/delete`);
      setCurrentWord(response.data.current_word || '[empty]');
    } catch (error) {
      console.error('Error deleting character:', error);
    }
  };

  const clearWord = async () => {
    if (window.confirm('Clear the entire word?')) {
      try {
        const response = await axios.post(`${API_URL}/api/word/clear`);
        setCurrentWord('[empty]');
      } catch (error) {
        console.error('Error clearing word:', error);
      }
    }
  };

  const resetSystem = async () => {
    if (window.confirm('Reset everything?')) {
      try {
        await axios.post(`${API_URL}/api/word/reset`);
        setCurrentWord('[empty]');
        setCurrentLetter('-');
        setConfidence(0);
        setHandCount(0);
        setCountdown(5.0);
      } catch (error) {
        console.error('Error resetting:', error);
      }
    }
  };

  // Get confidence color
  const getConfidenceClass = () => {
    if (confidence >= 0.7) return 'confidence-high';
    if (confidence >= 0.5) return 'confidence-medium';
    return 'confidence-low';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center p-5 font-sans">
      <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-4xl w-full animate-fadeIn">
  
        {/* Header */}
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-2">
          🤟 Hand Gesture Recognition
        </h1>
        <p className="text-center text-gray-500 mb-8">
          Show hand gestures to build words letter by letter
        </p>
  
        {/* Webcam */}
        <div className="relative bg-black rounded-xl overflow-hidden mb-6 min-h-[480px] flex items-center justify-center">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full transform -scale-x-100 ${isWebcamActive ? 'block' : 'hidden'}`}
          />
  
          {!isWebcamActive && (
            <div className="text-white text-center">
              <p className="text-lg mb-2">Click "Start Webcam" to begin</p>
              <p className="text-sm text-gray-300">Your webcam will open in the browser</p>
            </div>
          )}
  
          <canvas ref={canvasRef} className="hidden" />
        </div>
  
        {/* Status Bar */}
        <div className="bg-gray-100 rounded-lg p-4 mb-6 space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Current Prediction:</span>
            <span className="text-4xl font-bold text-indigo-500">{currentLetter}</span>
          </div>
  
          <div className="flex justify-between">
            <span className="text-gray-600">Confidence:</span>
            <span
              className={`font-bold ${
                confidence >= 0.7
                  ? 'text-green-600'
                  : confidence >= 0.5
                  ? 'text-yellow-500'
                  : 'text-red-600'
              }`}
            >
              {(confidence * 100).toFixed(0)}%
            </span>
          </div>
  
          <div className="flex justify-between">
            <span className="text-gray-600">Hands Detected:</span>
            <span className="font-bold text-gray-800">{handCount}</span>
          </div>
  
          <div className="text-center text-yellow-500 font-bold text-lg">
            Next add in: {countdown}s
          </div>
        </div>
  
        {/* Word Display */}
        <div className="bg-indigo-500 text-white rounded-lg text-center text-3xl font-bold py-5 mb-6 min-h-[80px] flex items-center justify-center break-all">
          {currentWord}
        </div>
  
        {/* Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 mb-6">
          <button
            onClick={isWebcamActive ? stopWebcam : startWebcam}
            disabled={backendStatus !== 'connected'}
            className={`col-span-full py-3 rounded-lg font-semibold text-white transition transform hover:-translate-y-1 ${
              isWebcamActive
                ? 'bg-red-600 hover:bg-red-700'
                : 'bg-indigo-500 hover:bg-indigo-600'
            } disabled:opacity-50`}
          >
            {isWebcamActive ? 'Stop Webcam' : 'Start Webcam'}
          </button>
  
          <button onClick={addSpace} className="btn bg-green-600">
            Add Space
          </button>
  
          <button onClick={deleteChar} className="btn bg-yellow-400 text-gray-800">
            Delete Last
          </button>
  
          <button onClick={clearWord} className="btn bg-gray-600">
            Clear Word
          </button>
  
          <button onClick={resetSystem} className="btn bg-cyan-600 col-span-full">
            Reset
          </button>
        </div>
  
        {/* Instructions */}
        <div className="bg-blue-50 border-l-4 border-indigo-500 p-4 rounded mb-4 text-sm">
          <h3 className="font-semibold text-indigo-600 mb-2">📋 Instructions:</h3>
          <ul className="list-disc ml-5 space-y-1 text-gray-700">
            <li>Click "Start Webcam" to begin</li>
            <li>Show a hand gesture and hold it steady</li>
            <li>After 5 seconds, the detected letter is added</li>
            <li>Use buttons to manage the word</li>
            <li>Confidence must be above 50%</li>
          </ul>
        </div>
  
        {/* Connection Status */}
        <div
          className={`text-center font-semibold text-sm py-2 rounded-lg ${
            backendStatus === 'connected'
              ? 'bg-green-100 text-green-700'
              : backendStatus === 'error'
              ? 'bg-red-100 text-red-700'
              : 'bg-yellow-100 text-yellow-700'
          }`}
        >
          {statusMessage}
        </div>
      </div>
    </div>
  );
  

}

export default App;