import React from 'react';

const StatusPanel = ({ currentLetter, confidence, handCount, countdown }) => {
  return (
    <div className="grid grid-cols-2 gap-3 mb-4 text-sm">
      <div className="bg-gray-100 p-3 rounded-lg">
        <span className="text-gray-600">Letter:</span> 
        <b className="text-2xl ml-2">{currentLetter}</b>
      </div>
      <div className="bg-gray-100 p-3 rounded-lg">
        <span className="text-gray-600">Confidence:</span> 
        <b className="text-xl ml-2">{(confidence * 100).toFixed(0)}%</b>
      </div>
      <div className="bg-gray-100 p-3 rounded-lg">
        <span className="text-gray-600">Hands Detected:</span> 
        <b className="text-xl ml-2">{handCount}</b>
      </div>
      <div className="bg-gray-100 p-3 rounded-lg">
        <span className="text-gray-600">Next Letter:</span> 
        <b className="text-xl ml-2">{countdown.toFixed(1)}s</b>
      </div>
    </div>
  );
};

export default StatusPanel;