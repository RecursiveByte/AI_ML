import React from 'react';

const WordControls = ({ 
  isWebcamActive, 
  currentWord, 
  onAddSpace, 
  onDeleteLetter, 
  onClearWord 
}) => {
  return (
    <div className="grid grid-cols-3 gap-3 mb-4">
      <button
        onClick={onAddSpace}
        disabled={!isWebcamActive}
        className="py-3 rounded-lg font-semibold bg-blue-500 hover:bg-blue-600 text-white disabled:bg-gray-300 disabled:cursor-not-allowed transition-all"
      >
        ␣ Space
      </button>
      <button
        onClick={onDeleteLetter}
        disabled={!isWebcamActive || currentWord.length === 0}
        className="py-3 rounded-lg font-semibold bg-yellow-500 hover:bg-yellow-600 text-white disabled:bg-gray-300 disabled:cursor-not-allowed transition-all"
      >
        ⌫ Delete
      </button>
      <button
        onClick={onClearWord}
        disabled={!isWebcamActive}
        className="py-3 rounded-lg font-semibold bg-orange-500 hover:bg-orange-600 text-white disabled:bg-gray-300 disabled:cursor-not-allowed transition-all"
      >
        🗑 Clear
      </button>
    </div>
  );
};

export default WordControls;