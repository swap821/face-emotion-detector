"""
emotion_labels.py — Emotion Metadata

Maps emotion indices to human-readable information.
7 emotions from FER-2013 dataset.
"""

EMOTIONS = {
    0: {'name': 'angry',     'emoji': '😠', 'color': '#ef4444', 'description': 'Furrowed brows, tightened jaw, glaring eyes'},
    1: {'name': 'disgust',   'emoji': '🤢', 'color': '#22c55e', 'description': 'Wrinkled nose, raised upper lip, narrowed eyes'},
    2: {'name': 'fear',      'emoji': '😨', 'color': '#8b5cf6', 'description': 'Wide eyes, raised eyebrows, tense mouth'},
    3: {'name': 'happy',     'emoji': '😊', 'color': '#10b981', 'description': 'Smiling mouth, raised cheeks, crow\'s feet'},
    4: {'name': 'sad',       'emoji': '😢', 'color': '#3b82f6', 'description': 'Down-turned mouth, drooping eyelids'},
    5: {'name': 'surprise',  'emoji': '😲', 'color': '#f59e0b', 'description': 'Wide eyes, raised eyebrows, open mouth'},
    6: {'name': 'neutral',   'emoji': '😐', 'color': '#6b7280', 'description': 'Relaxed face, no strong expression'},
}


def get_emotion_info(index):
    """Get info for an emotion by index."""
    return EMOTIONS.get(index, EMOTIONS[6])


def get_emotion_color(index):
    """Get color for an emotion by index."""
    return EMOTIONS.get(index, EMOTIONS[6])['color']