import React from 'react';

const VideoCanvas = ({ videoRef, canvasRef, isWebcamActive }) => {
  return (
    <div className="relative bg-black rounded-xl overflow-hidden mb-6" style={{height: '480px'}}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          transform: 'scaleX(-1)',
          display: isWebcamActive ? 'block' : 'none'
        }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          display: isWebcamActive ? 'block' : 'none'
        }}
      />

      {!isWebcamActive && (
        <div className="flex items-center justify-center h-full text-white text-lg">
          Click "Start Webcam" to begin
        </div>
      )}
    </div>
  );
};

export default VideoCanvas;