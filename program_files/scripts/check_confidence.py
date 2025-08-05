#!/usr/bin/env python3
from database.enhanced_conversation_db import EnhancedConversationDB

db = EnhancedConversationDB()
features, metadata = db.get_data('audio_features', return_features=True)

print('GMM Confidence Analysis:')
confidences = []
for i, meta in enumerate(metadata):
    conf = meta.get('gmm_confidence', 'None')
    speaker = meta.get('gmm_speaker', 'None')
    orig_speaker = meta.get('speaker', 'Unknown')
    if conf != 'None':
        confidences.append(float(conf))
        if i < 5:
            print(f'Sample {i}: {orig_speaker} -> {speaker} (conf: {conf:.3f})')

if confidences:
    print(f'\nConfidence stats:')
    print(f'  Min: {min(confidences):.3f}')
    print(f'  Max: {max(confidences):.3f}')
    print(f'  Avg: {sum(confidences)/len(confidences):.3f}')
    print(f'  Count < 1.0: {sum(1 for c in confidences if c < 1.0)}')
else:
    print('No GMM confidences found')