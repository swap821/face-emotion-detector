import { useState, useRef, useEffect, useCallback } from 'react';
import io from 'socket.io-client';

/**
 * WebcamView.jsx — Real-Time Webcam with Emotion Overlay
 * 
 * Uses getUserMedia to access webcam and Socket.IO to stream
 * frames to the backend for real-time emotion detection.
 */

const WebcamView = ({ onEmotionUpdate }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState(null);
  const socketRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      videoRef.current.play();
      setIsActive(true);
      setError(null);

      // Connect WebSocket
      socketRef.current = io('http://localhost:5000');
      socketRef.current.on('prediction', (data) => {
        onEmotionUpdate(data);
        drawOverlay(data);
      });

      // Send frames at 5 FPS
      intervalRef.current = setInterval(() => {
        captureAndSend();
      }, 200);
    } catch (err) {
      setError('Camera access denied. Please allow camera permissions.');
    }
  }, [onEmotionUpdate]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
    }
    if (socketRef.current) {
      socketRef.current.disconnect();
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    setIsActive(false);
  }, []);

  const captureAndSend = () => {
    if (!videoRef.current || !socketRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0);
    const base64 = canvas.toDataURL('image/jpeg', 0.7);
    socketRef.current.emit('frame', base64);
  };

  const drawOverlay = (data) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    ctx.drawImage(video, 0, 0);
    
    if (data.results) {
      data.results.forEach(r => {
        const { x, y, w, h } = r.box;
        ctx.strokeStyle = '#4d7dff';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = '#4d7dff';
        ctx.font = 'bold 20px Inter';
        ctx.fillText(`${r.emoji} ${r.emotion}`, x, y - 10);
      });
    }
  };

  return (
    <div className="bg-[#10121a] border border-white/10 rounded-2xl p-4">
      <div className="relative">
        <video ref={videoRef} className="w-full rounded-xl" style={{ display: isActive ? 'none' : 'block' }} />
        <canvas ref={canvasRef} className="w-full rounded-xl" style={{ display: isActive ? 'block' : 'none' }} />
        
        {!isActive && (
          <div className="w-full h-64 bg-[#0a0b10] rounded-xl flex items-center justify-center">
            <p className="text-gray-500">Camera off</p>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
          {error}
        </div>
      )}

      <button
        onClick={isActive ? stopCamera : startCamera}
        className={`mt-4 w-full py-3 rounded-xl font-bold transition-all ${
          isActive
            ? 'bg-red-600 hover:bg-red-500 text-white'
            : 'bg-blue-600 hover:bg-blue-500 text-white'
        }`}
      >
        {isActive ? 'Stop Camera' : 'Start Camera'}
      </button>
    </div>
  );
};

export default WebcamView;