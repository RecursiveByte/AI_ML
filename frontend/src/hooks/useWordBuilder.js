import { useState, useEffect, useRef } from 'react';
import { LETTER_ADD_INTERVAL, COUNTDOWN_INTERVAL } from '../config/constants';

export const useWordBuilder = (isWebcamActive, currentLetter) => {
  const [currentWord, setCurrentWord] = useState("");
  const [countdown, setCountdown] = useState(5.0);
  const [lastAddedLetter, setLastAddedLetter] = useState("");
  
  const letterAddTimerRef = useRef(null);
  const lastLetterRef = useRef("-");
  const countdownIntervalRef = useRef(null);

  // Update lastLetterRef when currentLetter changes
  useEffect(() => {
    lastLetterRef.current = currentLetter;
  }, [currentLetter]);

  // Word building logic
  useEffect(() => {
    if (!isWebcamActive) return;

    // Start countdown timer
    countdownIntervalRef.current = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 0.1) return 5.0;
        return prev - 0.1;
      });
    }, COUNTDOWN_INTERVAL);

    // Start letter adding timer
    letterAddTimerRef.current = setInterval(() => {
      addLetterToWord();
    }, LETTER_ADD_INTERVAL);

    return () => {
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current);
      }
      if (letterAddTimerRef.current) {
        clearInterval(letterAddTimerRef.current);
      }
    };
  }, [isWebcamActive]);

  const addLetterToWord = () => {
    const letter = lastLetterRef.current;
    
    // Don't add if no valid letter detected
    if (!letter || letter === "-" || letter === "none") {
      return;
    }

    // Add letter to word
    setCurrentWord(prev => prev + letter);
    setLastAddedLetter(letter);
    
    // Reset countdown
    setCountdown(5.0);
  };

  const clearWord = () => {
    setCurrentWord("");
    setLastAddedLetter("");
    setCountdown(5.0);
  };

  const deleteLastLetter = () => {
    setCurrentWord(prev => prev.slice(0, -1));
  };

  const addSpace = () => {
    setCurrentWord(prev => prev + " ");
  };

  return {
    currentWord,
    countdown,
    lastAddedLetter,
    clearWord,
    deleteLastLetter,
    addSpace
  };
};