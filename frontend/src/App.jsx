import React, { useState, useEffect, useRef } from "react";
import VideoCanvas from "./components/VideoCanvas";
import StatusPanel from "./components/StatusPanel";
import WordDisplay from "./components/WordDisplay";
import WordControls from "./components/WordControls";
import WebcamControls from "./components/WebcamControls";
import ColorLegend from "./components/ColorLegend";
import { useWebcam } from "./hooks/useWebcam";
import { useWordBuilder } from "./hooks/useWordBuilder";
import { useBackendConnection } from "./hooks/useBackendConnection";
import { API_URL } from "./config/constants";

function App() {
  // Backend connection state
  const { backendStatus, statusMessage, checkBackendConnection } =
    useBackendConnection(API_URL);

  // Webcam and prediction state
  const {
    isWebcamActive,
    videoRef,
    canvasRef,
    currentLetter,
    confidence,
    handCount,
    startWebcam,
    stopWebcam,
  } = useWebcam(API_URL, backendStatus);

  // Word building state
  const { currentWord, countdown, clearWord, deleteLastLetter, addSpace } =
    useWordBuilder(isWebcamActive, currentLetter);

  // Check backend on mount
  useEffect(() => {
    checkBackendConnection();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center p-5">
      <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-4xl w-full">
        <h1 className="text-3xl font-bold text-center mb-6">ISL-Vision</h1>
        <p className="text-center text-lg text-gray-800 mb-6">
          Indian Sign Language Recognition
        </p>

        <VideoCanvas
          videoRef={videoRef}
          canvasRef={canvasRef}
          isWebcamActive={isWebcamActive}
        />

        <StatusPanel
          currentLetter={currentLetter}
          confidence={confidence}
          handCount={handCount}
          countdown={countdown}
        />

        <WordDisplay currentWord={currentWord} />

        <WordControls
          isWebcamActive={isWebcamActive}
          currentWord={currentWord}
          onAddSpace={addSpace}
          onDeleteLetter={deleteLastLetter}
          onClearWord={clearWord}
        />

        <WebcamControls
          isWebcamActive={isWebcamActive}
          backendStatus={backendStatus}
          statusMessage={statusMessage}
          onStart={startWebcam}
          onStop={stopWebcam}
        />

        <ColorLegend />
      </div>
    </div>
  );
}

export default App;
