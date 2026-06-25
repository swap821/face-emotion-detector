const EMOTIONS = [
  { name: 'happy', emoji: '😊', color: '#10b981', desc: 'Smiling, raised cheeks' },
  { name: 'sad', emoji: '😢', color: '#3b82f6', desc: 'Down-turned mouth' },
  { name: 'angry', emoji: '😠', color: '#ef4444', desc: 'Furrowed brows' },
  { name: 'surprise', emoji: '😲', color: '#f59e0b', desc: 'Wide eyes' },
  { name: 'fear', emoji: '😨', color: '#8b5cf6', desc: 'Tense, wide eyes' },
  { name: 'disgust', emoji: '🤢', color: '#22c55e', desc: 'Wrinkled nose' },
  { name: 'neutral', emoji: '😐', color: '#6b7280', desc: 'Relaxed face' },
];

const EmotionLegend = ({ currentEmotion, confidence }) => (
  <div className="bg-[#10121a] border border-white/10 rounded-2xl p-6 space-y-3">
    <h3 className="font-bold text-lg mb-4">Emotions</h3>
    
    {EMOTIONS.map((e) => (
      <div
        key={e.name}
        className={`flex items-center gap-3 p-3 rounded-xl transition-all ${
          currentEmotion === e.name ? 'bg-white/10 border border-white/20' : ''
        }`}
      >
        <span className="text-2xl">{e.emoji}</span>
        <div className="flex-1">
          <p className="font-medium capitalize">{e.name}</p>
          <p className="text-xs text-gray-500">{e.desc}</p>
        </div>
        {currentEmotion === e.name && (
          <span className="text-sm font-bold" style={{ color: e.color }}>
            {(confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
    ))}
  </div>
);

export default EmotionLegend;