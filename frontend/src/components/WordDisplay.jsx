import React from 'react';

const WordDisplay = ({ currentWord }) => {
  return (
    <div className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg text-center text-4xl font-bold py-6 mb-4 shadow-lg min-h-[80px] flex items-center justify-center">
      {currentWord || "[empty]"}
    </div>
  );
};

export default WordDisplay;