import { HAND_CONNECTIONS, LANDMARK_COLORS, HAND_CONNECTION_COLORS } from '../config/constants';

export const drawHandLandmarks = (canvas, results, video) => {
  if (!canvas || !video) return;
  
  const ctx = canvas.getContext("2d");

  if (!video.videoWidth || !video.videoHeight) return;

  // Set canvas to match video dimensions exactly
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Save context state
  ctx.save();
  
  // Mirror the canvas to match the mirrored video
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    // First pass: Draw all connection lines for all hands
    drawConnections(ctx, results.multiHandLandmarks, canvas);
    
    ctx.globalAlpha = 1.0; // Reset alpha
    
    // Second pass: Draw all landmark points on top of all lines
    drawLandmarkPoints(ctx, results.multiHandLandmarks, canvas);
  }
  
  // Restore context state
  ctx.restore();
};

const drawConnections = (ctx, hands, canvas) => {
  ctx.lineWidth = 6;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  
  for (let handIdx = 0; handIdx < hands.length; handIdx++) {
    const landmarks = hands[handIdx];
    
    for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
      const start = landmarks[startIdx];
      const end = landmarks[endIdx];
      
      if (!start || !end) continue;
      
      const startX = start.x * canvas.width;
      const startY = start.y * canvas.height;
      const endX = end.x * canvas.width;
      const endY = end.y * canvas.height;
      
      // Different colors for each hand to distinguish them
      ctx.strokeStyle = handIdx === 0 
        ? HAND_CONNECTION_COLORS.hand1 
        : HAND_CONNECTION_COLORS.hand2;
      
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    }
  }
};

const drawLandmarkPoints = (ctx, hands, canvas) => {
  for (let handIdx = 0; handIdx < hands.length; handIdx++) {
    const landmarks = hands[handIdx];
    
    landmarks.forEach((landmark, idx) => {
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;
      
      // Draw outer circle (border)
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, 2 * Math.PI);
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();
      
      // Add shadow for depth
      ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
      ctx.shadowBlur = 4;
      ctx.shadowOffsetX = 2;
      ctx.shadowOffsetY = 2;
      
      // Draw inner filled circle
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = getLandmarkColor(idx);
      ctx.fill();
      
      // Reset shadow
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
      
      // Add hand number label for wrist
      if (idx === 0) {
        ctx.font = 'bold 16px Arial';
        ctx.fillStyle = '#FFFFFF';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 3;
        ctx.strokeText(`H${handIdx + 1}`, x + 15, y);
        ctx.fillText(`H${handIdx + 1}`, x + 15, y);
      }
    });
  }
};

const getLandmarkColor = (idx) => {
  if (idx === 0) return LANDMARK_COLORS.wrist;
  if (idx >= 1 && idx <= 4) return LANDMARK_COLORS.thumb;
  if (idx >= 5 && idx <= 8) return LANDMARK_COLORS.index;
  if (idx >= 9 && idx <= 12) return LANDMARK_COLORS.middle;
  if (idx >= 13 && idx <= 16) return LANDMARK_COLORS.ring;
  if (idx >= 17 && idx <= 20) return LANDMARK_COLORS.pinky;
  return '#FFFFFF';
};