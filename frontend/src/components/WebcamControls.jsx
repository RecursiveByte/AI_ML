import React from 'react';

const WebcamControls = ({ 
  isWebcamActive, 
  backendStatus, 
  statusMessage, 
  onStart, 
  onStop 
}) => {
  return (
    <>
      <button
        onClick={isWebcamActive ? onStop : onStart}
        disabled={backendStatus !== "connected"}
        className={`w-full py-4 rounded-lg font-semibold text-white text-lg transition-all ${
          isWebcamActive 
            ? "bg-red-600 hover:bg-red-700 shadow-lg" 
            : backendStatus === "connected"
              ? "bg-indigo-600 hover:bg-indigo-700 shadow-lg"
              : "bg-gray-400 cursor-not-allowed"
        }`}
      >
        {isWebcamActive ? "⏹ Stop Webcam" : "▶ Start Webcam"}
      </button>

      <div className={`mt-4 text-center text-sm font-semibold ${
        backendStatus === "connected" ? "text-green-600" : "text-red-600"
      }`}>
        ● {statusMessage}
      </div>
    </>
  );
};

export default WebcamControls;