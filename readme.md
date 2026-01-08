#Version required

pip install tensorflow==2.19.0
pip install mediapipe==0.10.14
pip install protobuf==4.25.3

"""
═══════════════════════════════════════════════════════════════════════════
    HAND GESTURE RECOGNITION WITH DNN - COMPLETE MASTER COURSE
    From Zero to Hero: Understanding Every Single Step
═══════════════════════════════════════════════════════════════════════════

Author: Your AI Master Teacher
Purpose: Build a real hand gesture recognition system with deep understanding
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("="*80)
print("SECTION 1: THE BIG PICTURE - What Are We Building?")
print("="*80)

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    THE COMPLETE PIPELINE                               ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Step 1: CAPTURE CAMERA → Get video frames                            ║
║           ↓                                                            ║
║  Step 2: DETECT HAND → Find hand in frame (MediaPipe)                 ║
║           ↓                                                            ║
║  Step 3: EXTRACT LANDMARKS → Get 21 hand keypoints (x, y, z)          ║
║           ↓                                                            ║
║  Step 4: NORMALIZE DATA → Make it independent of position/scale       ║
║           ↓                                                            ║
║  Step 5: TRAIN DNN → Teach network to recognize gestures              ║
║           ↓                                                            ║
║  Step 6: PREDICT → Real-time gesture recognition!                     ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

STORY TIME: Imagine teaching a child to recognize hand gestures
──────────────────────────────────────────────────────────────────

You show them different hand shapes (👍, ✌️, ✊) and say:
"This is thumbs up, this is peace sign, this is a fist"

The child learns by:
1. SEEING the hand (camera)
2. IDENTIFYING key points (thumb tip, finger tips - landmarks)
3. UNDERSTANDING relationships (thumb is UP, fingers are DOWN)
4. REMEMBERING patterns (brain learning - DNN training)
5. RECOGNIZING new gestures (prediction)

Our AI does EXACTLY the same thing!
""")

print("\n" + "="*80)
print("SECTION 2: UNDERSTANDING MEDIAPIPE - The Hand Detector")
print("="*80)

print("""
WHAT IS MEDIAPIPE?
─────────────────────────────────────────────────────────────────────────
MediaPipe is like having a super-smart assistant who can instantly find
and mark 21 key points on your hand.

Think of it like this:
• You show a photo to a friend
• Friend instantly points: "There's the thumb tip! There's the pinky!"
• Friend does this 30 times per second (real-time!)

THE 21 HAND LANDMARKS:
─────────────────────────────────────────────────────────────────────────
       
         8  12  16  20        ← Finger tips
         │  │   │   │
         7  11  15  19        ← Finger joints (top)
         │  │   │   │
         6  10  14  18        ← Finger joints (middle)
         │  │   │   │
         5  9   13  17        ← Finger base
          ╲ │   │  ╱
           4│   │ ╱           ← Thumb
            3  2╱
             ╲│╱
              1               ← Wrist base
              │
              0               ← Wrist center

Each landmark has 3 coordinates:
• X: Left-right position (0.0 to 1.0)
• Y: Up-down position (0.0 to 1.0)  
• Z: Depth/distance from camera (relative)

Total features: 21 landmarks × 3 coordinates = 63 numbers!
""")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print("\n✓ MediaPipe Hands initialized successfully!")
print("  • max_num_hands=1: We track ONE hand at a time")
print("  • min_detection_confidence=0.7: 70% sure it's a hand before detecting")
print("  • min_tracking_confidence=0.7: 70% sure we're still tracking same hand")

print("\n" + "="*80)
print("SECTION 3: DATA COLLECTION - Teaching the AI")
print("="*80)

print("""
THE TEACHING PROCESS:
─────────────────────────────────────────────────────────────────────────

Just like teaching a child, we need to show MANY examples:

Gesture 0 (Fist):        Show 100 examples → AI learns "fingers curled"
Gesture 1 (Thumbs up):   Show 100 examples → AI learns "thumb up, fingers down"
Gesture 2 (Peace):       Show 100 examples → AI learns "2 fingers up"
... and so on

WHY 100 EXAMPLES?
• More examples = Better learning
• Different angles, distances, lighting
• AI sees variations and learns the CORE pattern

DATA STRUCTURE:
─────────────────────────────────────────────────────────────────────────
After collection, we'll have:

X (features):           Shape: (1000, 63)
  [                     1000 examples, each with 63 numbers
    [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21],  ← Example 1
    [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21],  ← Example 2
    ...
  ]

y (labels):             Shape: (1000,)
  [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
   └─100x─┘    └─100x─┘    └─100x─┘
   Gesture 0   Gesture 1   Gesture 2
""")

class GestureDataCollector:
    """
    This class helps us collect hand gesture data
    Think of it as your data collection assistant!
    """
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_landmarks(self, image):
        """
        Extract 21 hand landmarks from an image
        
        INPUT: Image from camera (RGB format)
        OUTPUT: Array of 63 numbers [x1,y1,z1, x2,y2,z2, ..., x21,y21,z21]
        
        STORY: Like asking "Where are all the fingers and joints?"
        """
        # Convert BGR to RGB (OpenCV uses BGR, MediaPipe uses RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        # If hand is detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand
            
            # Extract all 21 landmarks (x, y, z)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        
        return None
    
    def collect_data_for_gesture(self, gesture_name, gesture_label, num_samples=100):
        """
        Collect training data for ONE gesture
        
        PARAMETERS:
        • gesture_name: Name like "thumbs_up" (for display)
        • gesture_label: Number like 0, 1, 2 (for AI)
        • num_samples: How many examples to collect (default 100)
        
        STORY: Like taking 100 photos of someone making thumbs up
        """
        print(f"\n{'='*60}")
        print(f"Collecting data for: {gesture_name} (Label: {gesture_label})")
        print(f"{'='*60}")
        print(f"We'll collect {num_samples} examples")
        print(f"Press 's' to start collecting, 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        collected_data = []
        labels = []
        count = 0
        collecting = False
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror image
            
            # Try to extract landmarks
            landmarks = self.extract_landmarks(frame)
            
            # Draw status on frame
            status_color = (0, 255, 0) if collecting else (0, 0, 255)
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Collected: {count}/{num_samples}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(frame, f"Press 's' to start", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # If collecting and hand detected
            if collecting and landmarks is not None:
                collected_data.append(landmarks)
                labels.append(gesture_label)
                count += 1
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                collecting = True
                print("Started collecting!")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✓ Collected {count} examples for {gesture_name}")
        return np.array(collected_data), np.array(labels)

print("\n✓ Data Collector class ready!")
print("\nNOTE: To collect data, you would run:")
print("  collector = GestureDataCollector()")
print("  X, y = collector.collect_data_for_gesture('thumbs_up', 0, 100)")

print("\n" + "="*80)
print("SECTION 4: DATA NORMALIZATION - Making Data Universal")
print("="*80)

print("""
WHY NORMALIZE?
─────────────────────────────────────────────────────────────────────────

PROBLEM: Raw landmark coordinates depend on:
• Where your hand is in the frame (left/right/center)
• How close you are to camera (near/far)
• Camera resolution

Example:
Hand close to camera:    x=0.8, y=0.6
Same hand, far away:     x=0.5, y=0.5
^ Same gesture, different numbers! AI gets confused! 😵

SOLUTION: Normalize relative to WRIST (landmark 0)
─────────────────────────────────────────────────────────────────────────

STORY: Imagine measuring your friend's height
• Don't say "5 feet from the ground" (depends where they stand)
• Say "5 feet tall" (relative to their feet) ✓

Similarly:
• Don't say "thumb at position 0.8" (depends on hand location)
• Say "thumb 0.3 units above wrist" (relative to wrist) ✓

MATHEMATICAL TRANSFORMATION:
─────────────────────────────────────────────────────────────────────────
Original: [x0, y0, z0, x1, y1, z1, ..., x21, y21, z21]
           └─wrist─┘  └─thumb─┘

Normalized: All landmarks MINUS wrist position
x1_new = x1 - x0  (thumb X relative to wrist X)
y1_new = y1 - y0  (thumb Y relative to wrist Y)
z1_new = z1 - z0  (thumb Z relative to wrist Z)

Result: Gesture looks the same regardless of position! 🎯
""")

def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to wrist (landmark 0)
    
    INPUT: [x0, y0, z0, x1, y1, z1, ..., x21, y21, z21]
    OUTPUT: Normalized values relative to wrist
    
    ANALOGY: Converting "position in room" to "position relative to person"
    """
    landmarks = landmarks.copy()
    
    # Extract wrist position (first landmark)
    wrist_x = landmarks[0]
    wrist_y = landmarks[1]
    wrist_z = landmarks[2]
    
    # Subtract wrist from all landmarks
    for i in range(0, len(landmarks), 3):
        landmarks[i] -= wrist_x      # Normalize X
        landmarks[i+1] -= wrist_y    # Normalize Y
        landmarks[i+2] -= wrist_z    # Normalize Z
    
    return landmarks

# Test normalization
test_landmarks = np.random.rand(63) * 0.5  # Random landmarks
normalized = normalize_landmarks(test_landmarks)

print("\n✓ Normalization function ready!")
print(f"\nExample transformation:")
print(f"Original wrist position: ({test_landmarks[0]:.3f}, {test_landmarks[1]:.3f}, {test_landmarks[2]:.3f})")
print(f"Normalized wrist position: ({normalized[0]:.3f}, {normalized[1]:.3f}, {normalized[2]:.3f})")
print(f"^ Notice: Wrist is now at (0, 0, 0) - our reference point!")

print("\n" + "="*80)
print("SECTION 5: BUILDING THE DNN - The Brain")
print("="*80)

print("""
THE NEURAL NETWORK ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────

Input Layer (63)         ← Hand landmark coordinates
     ↓
Dense Layer (128, relu)  ← First hidden layer: learns basic patterns
     ↓                     "thumb position", "finger angles"
Dropout (0.3)            ← Randomly drop 30% neurons (prevents overfitting)
     ↓
Dense Layer (64, relu)   ← Second hidden layer: combines patterns
     ↓                     "thumb up + fingers down = thumbs up"
Dropout (0.3)            ← More dropout
     ↓
Dense Layer (32, relu)   ← Third hidden layer: refines understanding
     ↓
Output Layer (10, softmax) ← 10 gestures with probabilities

LAYER-BY-LAYER EXPLANATION:
─────────────────────────────────────────────────────────────────────────

1. INPUT LAYER (63 neurons)
   • Takes 63 numbers (21 landmarks × 3 coordinates)
   • No processing, just passes data forward
   
2. DENSE(128, relu)
   • 128 neurons, each learning different pattern
   • ReLU activation: turns negatives to 0
   • Learns: "When thumb_x > 0.2 AND thumb_y < 0.3..."
   
3. DROPOUT(0.3)
   • During training, randomly ignores 30% of neurons
   • WHY? Prevents memorization, forces generalization
   • ANALOGY: Like studying with different friends each time
     (don't rely on one person's notes!)
   
4. DENSE(64, relu)
   • 64 neurons combining patterns from previous layer
   • Learns: "Pattern A + Pattern B = Specific gesture"
   
5. DENSE(32, relu)
   • 32 neurons for final refinement
   • Learns subtle differences between similar gestures
   
6. OUTPUT(10, softmax)
   • 10 neurons, one per gesture
   • Softmax: converts to probabilities that sum to 1
   • Output: [0.05, 0.02, 0.87, 0.01, ...]
              └── 87% confident it's gesture 2!

WHY THIS ARCHITECTURE?
─────────────────────────────────────────────────────────────────────────
• 128 → 64 → 32: Funnel shape is common (starts wide, narrows down)
• Wide layers early: capture many patterns
• Narrow layers later: focus on what matters
• Dropout: prevents overfitting (memorizing training data)
""")

def create_gesture_model(num_classes=10):
    """
    Create the DNN model for gesture recognition
    
    PARAMETERS:
    • num_classes: Number of different gestures (default 10)
    
    RETURNS:
    • Compiled Keras model ready for training
    """
    model = Sequential([
        # Input layer implicitly defined by first Dense layer
        Dense(128, activation='relu', input_shape=(63,), name='hidden_layer_1'),
        Dropout(0.3, name='dropout_1'),
        
        Dense(64, activation='relu', name='hidden_layer_2'),
        Dropout(0.3, name='dropout_2'),
        
        Dense(32, activation='relu', name='hidden_layer_3'),
        
        Dense(num_classes, activation='softmax', name='output_layer')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',      # Adam: Smart optimizer (adjusts learning rate)
        loss='categorical_crossentropy',  # For multi-class classification
        metrics=['accuracy']   # Track accuracy during training
    )
    
    return model

# Create and display model
model = create_gesture_model(num_classes=10)
print("\n✓ DNN Model created successfully!")
print("\nModel Architecture:")
model.summary()

print("\n" + "="*80)
print("SECTION 6: TRAINING THE MODEL - Learning Process")
print("="*80)

print("""
THE TRAINING PROCESS:
─────────────────────────────────────────────────────────────────────────

Training is like teaching through repetition and feedback:

1. FORWARD PASS:
   • Show the network a hand gesture
   • Network makes a prediction
   • Example: Sees thumbs up, predicts [0.1, 0.6, 0.2, 0.1, ...]
   
2. CALCULATE LOSS:
   • Compare prediction with correct answer
   • Correct: [0, 1, 0, 0, ...]  (gesture 1)
   • Predicted: [0.1, 0.6, 0.2, 0.1, ...]
   • Loss: How wrong is this? Higher = worse
   
3. BACKWARD PASS (Backpropagation):
   • Calculate: "Which weights caused this error?"
   • Adjust weights slightly to reduce error
   • ANALOGY: "Oh, I was wrong because I gave too much importance
               to finger position. Let me adjust that."
   
4. REPEAT:
   • Do this for ALL training examples
   • One complete cycle = 1 EPOCH
   • Repeat for many epochs (usually 50-200)

HYPERPARAMETERS EXPLAINED:
─────────────────────────────────────────────────────────────────────────

• EPOCHS: How many times to show all data
  - Too few: Underfitting (didn't learn enough)
  - Too many: Overfitting (memorized training data)
  - Sweet spot: Usually 50-100

• BATCH SIZE: How many examples to show before updating weights
  - Small (16-32): More updates, slower, better for small datasets
  - Large (128-256): Fewer updates, faster, needs more data
  - We use 32: Good balance

• VALIDATION SPLIT: % of data to test on (not used for training)
  - We use 0.2 = 20% for testing
  - Helps detect overfitting

WHAT TO WATCH DURING TRAINING:
─────────────────────────────────────────────────────────────────────────

Training Accuracy:    How well it learns training data
Validation Accuracy:  How well it works on NEW data

GOOD SIGNS:
✓ Both accuracies increase
✓ Both are close (within 5-10%)
✓ Smooth curves

BAD SIGNS:
✗ Training high (95%), Validation low (60%) → OVERFITTING!
✗ Both stuck at low values → Model too simple or bad data
✗ Wild fluctuations → Learning rate too high
""")

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Train the gesture recognition model
    
    PARAMETERS:
    • model: The DNN model to train
    • X_train: Training data (shape: num_samples, 63)
    • y_train: Labels (shape: num_samples, num_classes)
    • epochs: Number of training cycles
    • batch_size: Examples per weight update
    
    RETURNS:
    • history: Training history (loss, accuracy over time)
    """
    print(f"\nStarting training...")
    print(f"  • Training samples: {X_train.shape[0]}")
    print(f"  • Features per sample: {X_train.shape[1]}")
    print(f"  • Epochs: {epochs}")
    print(f"  • Batch size: {batch_size}")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # 20% for validation
        verbose=1  # Show progress
    )
    
    return history

print("\n✓ Training function ready!")

print("\n" + "="*80)
print("SECTION 7: MAKING PREDICTIONS - Using the Trained Model")
print("="*80)

print("""
PREDICTION PROCESS:
─────────────────────────────────────────────────────────────────────────

Once trained, prediction is simple and fast:

1. CAPTURE FRAME from camera
   ↓
2. DETECT HAND with MediaPipe
   ↓
3. EXTRACT 21 LANDMARKS (63 numbers)
   ↓
4. NORMALIZE landmarks (relative to wrist)
   ↓
5. FEED TO MODEL
   ↓
6. GET PROBABILITIES
   Output: [0.02, 0.05, 0.87, 0.01, 0.03, 0.01, 0.01, 0.00, 0.00, 0.00]
            └───────────────┬──────────────────────────────────────────┘
                    87% confident it's Gesture 2!
   ↓
7. PICK HIGHEST probability → Final prediction!

MODEL OUTPUT INTERPRETATION:
─────────────────────────────────────────────────────────────────────────

Softmax output always sums to 1.0 (100%):
[0.02, 0.05, 0.87, 0.01, 0.03, 0.01, 0.01, 0.00, 0.00, 0.00]
 │     │     │                                               │
 2%    5%    87%  ... ← These are CONFIDENCES, not counts!  0%

CONFIDENCE THRESHOLD:
─────────────────────────────────────────────────────────────────────────
• If max confidence < 0.6 (60%) → Don't predict (uncertain)
• If max confidence > 0.6 (60%) → Show prediction (confident)

WHY? Prevents false predictions when:
• Hand partially visible
• Between two gestures
• Unusual hand position
""")

def predict_gesture(model, landmarks, gesture_names, threshold=0.6):
    """
    Predict gesture from hand landmarks
    
    PARAMETERS:
    • model: Trained DNN model
    • landmarks: Array of 63 hand coordinates
    • gesture_names: List of gesture names ['fist', 'thumbs_up', ...]
    • threshold: Minimum confidence to show prediction (0.0 to 1.0)
    
    RETURNS:
    • gesture_name: Predicted gesture name (or 'Unknown')
    • confidence: Confidence score (0.0 to 1.0)
    • all_probabilities: Array of all class probabilities
    """
    # Normalize landmarks
    normalized = normalize_landmarks(landmarks)
    
    # Reshape for model input (model expects batch dimension)
    input_data = normalized.reshape(1, -1)
    
    # Get prediction
    probabilities = model.predict(input_data, verbose=0)[0]
    
    # Get class with highest probability
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    # Check threshold
    if confidence >= threshold:
        gesture_name = gesture_names[predicted_class]
    else:
        gesture_name = "Unknown"
    
    return gesture_name, confidence, probabilities

print("\n✓ Prediction function ready!")

print("\n" + "="*80)
print("SECTION 8: REAL-TIME GESTURE RECOGNITION")
print("="*80)

print("""
PUTTING IT ALL TOGETHER:
─────────────────────────────────────────────────────────────────────────

The real-time recognition loop:

while camera_is_on:
    1. Capture frame from camera
    2. Detect hand with MediaPipe
    3. If hand found:
        a. Extract 21 landmarks (63 numbers)
        b. Normalize relative to wrist
        c. Feed to trained model
        d. Get prediction + confidence
        e. Display on screen
    4. Show frame with prediction overlay
    5. If 'q' pressed: quit

PERFORMANCE TIPS:
─────────────────────────────────────────────────────────────────────────
• Model prediction is VERY fast (~1-2ms)
• MediaPipe detection is fast (~10-15ms)
• Total: 30-60 FPS easily achievable!
• Bottleneck: Camera capture, not AI
""")

def real_time_recognition(model, gesture_names):
    """
    Run real-time gesture recognition
    
    PARAMETERS:
    • model: Trained DNN model
    • gesture_names: List of gesture names
    
    CONTROLS:
    • 'q': Quit
    """
    print("\nStarting real-time recognition...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hand
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Draw hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
            
            # Extract and normalize landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks)
            
            # Predict gesture
            gesture, confidence, probs = predict_gesture(
                model, landmarks, gesture_names, threshold=0.6
            )
            
            # Display prediction
            if gesture != "Unknown":
                cv2.putText(frame, f"{gesture}: {confidence:.2f}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Unknown gesture", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

print("\n✓ Real-time recognition function ready!")

print("\n" + "="*80)
print("SECTION 9: SAVING AND LOADING MODELS")
print("="*80)

print("""
WHY SAVE MODELS?
─────────────────────────────────────────────────────────────────────────
• Training takes time (minutes to hours)
• Don't want to retrain every time!
• Save once, use forever

WHAT GETS SAVED?
─────────────────────────────────────────────────────────────────────────
• Model architecture (layers, neurons)
• Trained weights (all the learned parameters)
• Optimizer state
• Everything needed to make predictions!

FILE FORMATS:
─────────────────────────────────────────────────────────────────────────
• .h5: HDF5 format (older, widely supported)
• SavedModel: TensorFlow format (newer, recommended)
• .pkl: For data (not models)
""")

def save_model_and_config(model, gesture_names, filepath='gesture_model.h5'):
    """
    Save trained model and configuration
    
    PARAMETERS:
    • model: Trained Keras model
    • gesture_names: List of gesture names
    • filepath: Where to save model
    """
    # Save model
    model.save(filepath)
    print(f"✓ Model saved to {filepath}")
    
    # Save gesture names
    config_path = filepath.replace('.h5', '_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump({'gesture_names': gesture_names}, f)
    print(f"✓ Configuration saved to {config_path}")

def load_model_and_config(filepath='gesture_model.h5'):
    """
    Load trained model and configuration
    
    PARAMETERS:
    • filepath: Path to saved model
    
    RETURNS:
    • model: Loaded Keras model
    • gesture_names: List of gesture names
    """
    # Load model
    model = load_model(filepath)
    print(f"✓ Model loaded from {filepath}")
    
    