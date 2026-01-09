// API Configuration
export const API_URL = "https://ai-ml-bmmn.onrender.com";

// Timing Configuration
export const PREDICTION_INTERVAL = 1000; // predict every 1 sec
export const LETTER_ADD_INTERVAL = 5000; // add letter every 5 secs
export const COUNTDOWN_INTERVAL = 100; // countdown update interval

// MediaPipe Hand Connections - Exact mapping
export const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger  
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm connections
  [5, 9], [9, 13], [13, 17], [0, 17]
];

// Hand Landmark Colors
export const LANDMARK_COLORS = {
  wrist: '#FF0000',
  thumb: '#00FF00',
  index: '#0000FF',
  middle: '#FFFF00',
  ring: '#FF00FF',
  pinky: '#00FFFF'
};

// Hand Connection Colors
export const HAND_CONNECTION_COLORS = {
  hand1: '#00FF00',
  hand2: '#00CCFF'
};

// MediaPipe Configuration
export const MEDIAPIPE_CONFIG = {
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.3,
  minTrackingConfidence: 0.3
};

// Video Configuration
export const VIDEO_CONFIG = {
  width: { ideal: 1280 },
  height: { ideal: 720 },
  facingMode: "user"
};