import React from 'react';
import { LANDMARK_COLORS, HAND_CONNECTION_COLORS } from '../config/constants';

const ColorLegend = () => {
  const landmarkItems = [
    { color: LANDMARK_COLORS.wrist, label: 'Wrist' },
    { color: LANDMARK_COLORS.thumb, label: 'Thumb' },
    { color: LANDMARK_COLORS.index, label: 'Index' },
    { color: LANDMARK_COLORS.middle, label: 'Middle' },
    { color: LANDMARK_COLORS.ring, label: 'Ring' },
    { color: LANDMARK_COLORS.pinky, label: 'Pinky' }
  ];

  const connectionItems = [
    { color: HAND_CONNECTION_COLORS.hand1, label: 'Hand 1 (Green)' },
    { color: HAND_CONNECTION_COLORS.hand2, label: 'Hand 2 (Cyan)' }
  ];

  return (
    <div className="mt-6 p-4 bg-gray-50 rounded-lg border-2 border-gray-200">
      <h3 className="font-bold mb-3 text-sm text-gray-700">Hand Landmark Colors:</h3>
      <div className="grid grid-cols-3 gap-3 text-xs mb-4">
        {landmarkItems.map(({ color, label }) => (
          <div key={label} className="flex items-center gap-2">
            <div 
              className="w-4 h-4 rounded-full border-2 border-white" 
              style={{backgroundColor: color}}
            />
            <span className="font-medium">{label}</span>
          </div>
        ))}
      </div>
      <div className="border-t-2 border-gray-300 pt-3 mt-2">
        <h3 className="font-bold mb-2 text-sm text-gray-700">Hand Connections:</h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          {connectionItems.map(({ color, label }) => (
            <div key={label} className="flex items-center gap-2">
              <div 
                className="w-8 h-1 rounded" 
                style={{backgroundColor: color}}
              />
              <span className="font-medium">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ColorLegend;