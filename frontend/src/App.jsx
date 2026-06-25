import { useState, useRef, useCallback } from 'react';
import WebcamView from './components/WebcamView';
import EmotionLegend from './components/EmotionLegend';
import Footer from './components/Footer';

/**
 * App.jsx — Face Emotion Detector
 * 
 * Real-time webcam emotion detection with WebSocket streaming.
 */

function App() {
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const wsRef = useRef(null);

  const handleEmotionUpdate = useCallback((data) => {
    if (data.results && data.results.length > 0) {
      setEmotion(data.results[0]);
      setConfidence(data.results[0].confidence);
    }
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0b10] text-white">
      <nav className="border-b border-white/10 bg-[#10121a]/80 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-6 py-4 flex justify-between">
          <h1 className="text-xl font-bold">
            Emotion <span className="text-blue-400">Detector</span>
          </h1>
          <a href="https://github.com/swap821/face-emotion-detector" target="_blank" rel="noopener noreferrer" className="text-sm text-gray-400 hover:text-white">
            GitHub
          </a>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-extrabold mb-4">
            Real-Time{' '}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
              Emotion Detection
            </span>
          </h2>
          <p className="text-gray-400">CNN-powered facial expression recognition from your webcam</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <WebcamView onEmotionUpdate={handleEmotionUpdate} />
          </div>
          <div>
            <EmotionLegend currentEmotion={emotion?.emotion} confidence={confidence} />
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;