#sign language main wala
import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = './data'
MODEL_PATH = './model.p'
DATASET_PICKLE = './dataset.pickle'

ALPHABET = {i: chr(65 + i) for i in range(26)}  

NUM_CLASSES = 26
DATASET_SIZE = 100
DETECTION_CONFIDENCE = 0.5



def collect_images():
    """Collect hand gesture images from webcam - SIMPLE VERSION"""
    
    print("\n" + "="*60)
    print("STEP 1: COLLECTING GESTURE IMAGES")
    print("="*60)
    
    # Create data directories
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for i in range(NUM_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(i))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot access webcam!")
        return False
    
    for class_idx in range(NUM_CLASSES):
        print(f"\nCollecting for gesture: {ALPHABET[class_idx]}")
        print("Press 'Q' when ready, then it will collect 100 images...")
        
        
        ready = False
        while not ready:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Cannot read frame!")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            cv2.putText(frame, f"Gesture: {ALPHABET[class_idx]} - Press 'Q' to START", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow('Collecting Data', frame)
            
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                ready = True
        
        # Collect images
        counter = 0
        while counter < DATASET_SIZE:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save image
            img_path = os.path.join(DATA_DIR, str(class_idx), f'{counter}.jpg')
            cv2.imwrite(img_path, frame)
            
            
            cv2.putText(frame, f"Collecting: {counter + 1}/{DATASET_SIZE}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow('Collecting Data', frame)
            cv2.waitKey(25)
            
            counter += 1
        
        print(f"✓ Collected {DATASET_SIZE} images for {ALPHABET[class_idx]}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Data collection COMPLETE!")
    return True



def create_dataset():
    """Extract hand landmarks from images and create training dataset"""
    
    print("\n" + "="*60)
    print("STEP 2: CREATING DATASET (EXTRACTING LANDMARKS)")
    print("="*60)
    
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=DETECTION_CONFIDENCE
    )
    
    data = []
    labels = []
    skipped = 0
    processed = 0
    
    
    for class_idx in range(NUM_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(class_idx))
        
        if not os.path.exists(class_dir):
            print(f"ERROR: Directory not found: {class_dir}")
            continue
        
        print(f"Processing class {class_idx} ({ALPHABET[class_idx]})...")
        
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue
            
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks is None:
                skipped += 1
                continue
            
            
            landmarks_list = []
            x_coords = []
            y_coords = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)
            
            
            min_x = min(x_coords)
            min_y = min(y_coords)
            
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x = lm.x - min_x
                    y = lm.y - min_y
                    landmarks_list.append(x)
                    landmarks_list.append(y)
            
            
            while len(landmarks_list) < 42:
                landmarks_list.append(0)
            
            landmarks_list = landmarks_list[:42]
            
            data.append(landmarks_list)
            labels.append(class_idx)
            processed += 1
    
    hands.close()
    
    print(f"\n✓ Processed: {processed} images")
    print(f"✗ Skipped: {skipped} images (no hand detected)")
    
    
    dataset = {'data': np.array(data), 'labels': np.array(labels)}
    with open(DATASET_PICKLE, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"✓ Dataset saved to {DATASET_PICKLE}")
    return dataset



def train_model(dataset):
    """Train Random Forest classifier"""
    
    print("\n" + "="*60)
    print("STEP 3: TRAINING MODEL")
    print("="*60)
    
    X = dataset['data']
    y = dataset['labels']
    
    print(f"Training data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
    
    
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Training complete!")
    print(f"✓ Model Accuracy: {accuracy * 100:.2f}%")
    
  
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model}, f)
    
    print(f"✓ Model saved to {MODEL_PATH}")
    return model



def run_inference():
    """Real-time gesture recognition from webcam"""
    
    print("\n" + "="*60)
    print("STEP 4: RUNNING REAL-TIME INFERENCE")
    print("="*60)
    print("Press 'Q' to exit\n")
    
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
            model = model_dict['model']
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("ERROR: Model not found! Please train the model first (Step 3)")
        return
    
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=DETECTION_CONFIDENCE
    )
    
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot access webcam!")
        return
    
    print("Starting inference... (Press 'Q' to exit)")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Cannot read frame!")
            break
        
        H, W, C = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
            
            
            landmarks_list = []
            x_coords = []
            y_coords = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)
            
            min_x = min(x_coords)
            min_y = min(y_coords)
            
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x = lm.x - min_x
                    y = lm.y - min_y
                    landmarks_list.append(x)
                    landmarks_list.append(y)
            
            
            while len(landmarks_list) < 42:
                landmarks_list.append(0)
            landmarks_list = landmarks_list[:42]
            
            
            prediction = model.predict([np.array(landmarks_list)])
            predicted_char = ALPHABET[int(prediction[0])]
            
            
            x1 = int(min(x_coords) * W) - 20
            y1 = int(min(y_coords) * H) - 20
            x2 = int(max(x_coords) * W) + 20
            y2 = int(max(y_coords) * H) + 20
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, predicted_char, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        
        
        cv2.imshow('Sign Language Recognition', frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Inference stopped")

def main():
    """Main menu"""
    
    while True:
        print("\n" + "="*60)
        print("SIGN LANGUAGE DETECTION - MAIN MENU")
        print("="*60)
        print("1. Collect Training Images")
        print("2. Create Dataset (Extract Landmarks)")
        print("3. Train Model")
        print("4. Run Real-time Inference")
        print("5. Run Full Pipeline (1→2→3→4)")
        print("6. Exit")
        print("="*60)
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == '1':
            collect_images()
        
        elif choice == '2':
            if not os.path.exists(DATA_DIR):
                print("ERROR: No data directory found. Run Step 1 first!")
            else:
                dataset = create_dataset()
        
        elif choice == '3':
            if not os.path.exists(DATASET_PICKLE):
                print("ERROR: Dataset not found. Run Step 2 first!")
            else:
                with open(DATASET_PICKLE, 'rb') as f:
                    dataset = pickle.load(f)
                train_model(dataset)
        
        elif choice == '4':
            run_inference()
        
        elif choice == '5':
            print("\nRunning full pipeline (collecting → processing → training → inference)...")
            if collect_images():
                dataset = create_dataset()
                train_model(dataset)
                run_inference()
        
        elif choice == '6':
            print("Exiting... Goodbye!")
            break
        
        else:
            print("Invalid choice! Enter 1-6")



if __name__ == '__main__':
    main()
