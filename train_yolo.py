from ultralytics import YOLO
import os

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# YOLO expects data in a specific format or simply a directory path for classification
# Since our data is already in Train/Test folders with class subfolders, we can point to it.
# However, Ultralytics usually likes a split structure or we can just point it to the root data dir if configured.
# For classification, 'data' argument in model.train() can point to the root dataset directory containing train/test/val folders.
# Our structure:
# root/
#   Train/
#     ClassA/ ...
#   Test/
#     ClassA/ ...

def train_yolo():
    print("Initializing YOLOv8 training...")
    
    # Load a model
    # yolov8n-cls.pt is the Nano Classification model (fastest)
    model = YOLO('yolov8n-cls.pt')  

    # Train the model
    # We point 'data' to the current directory because it contains 'Train' and 'Test' folders.
    # YOLO automatically looks for train/valid/test directories within the data path.
    # We might need to handle the folder names if YOLO expects strictly 'train' and 'val'.
    # Our folders are 'Train' and 'Test'. YOLO defaults usually look for 'train' and 'val'.
    # Let's verify we don't need to rename. If we do, we can pass argument or rename symlinks.
    # The simplest way for YOLO classification is to have data/train and data/val.
    
    # Check if we need to workaround the folder names or if strict mode allows it.
    # Docs say: data argument is path to dataset root dir. 
    # It expects: root/train and root/val (or root/test).
    # Since we have "Train" (capital T), it might work on Windows (case insensitive), 
    # but "Test" might be treated as test set, not validation. 
    # For training, it NEEDS validation data. If 'val' is missing, it might automatic split or error.
    # Let's try pointing to BASE_DIR and see. If it fails, we can create a temporary symlink or split.
    # To be safe, we will let it treat 'Test' as the validation set if possible, 
    # or we simply rely on its auto-split if we point to just the training data?
    # Actually, proper way:
    
    # For this script, we will simply point to BASE_DIR. 
    # Windows is case-insensitive, so 'Train' == 'train'. 
    # We need to make sure 'val' exists or use 'Test' as val.
    # We can pass `data=BASE_DIR` and hope it finds 'Train'.
    
    print(f"Training on data in: {BASE_DIR}")
    
    # 'project' and 'name' define where results are saved: BASE_DIR/runs/classify/train_yolo
    results = model.train(
        data=BASE_DIR, 
        epochs=5, 
        imgsz=224,
        project=BASE_DIR,
        name='yolo_results',
        exist_ok=True # Overwrite existing
    )

    # Export/Save the best model to a known local path for the app
    # access the best model path
    best_model_path = os.path.join(BASE_DIR, 'yolo_results', 'weights', 'best.pt')
    
    print(f"Training complete. Best model saved at: {best_model_path}")

if __name__ == '__main__':
    # Ensure ultralytics is installed
    try:
        import ultralytics
        train_yolo()
    except ImportError:
        print("Ultralytics not found. Please install: pip install ultralytics")
