import onnxruntime as ort
import numpy as np
from PIL import Image
import os

def debug_onnx_classifier(image_path, model_path, labels_file):
    """
    Debug version of your ONNX classifier with additional checks
    """
    print(f"=== DEBUGGING ONNX CLASSIFIER ===")
    print(f"Image path: {image_path}")
    print(f"Model path: {model_path}")
    print(f"Labels file: {labels_file}")
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return
    if not os.path.exists(labels_file):
        print(f"ERROR: Labels file not found: {labels_file}")
        return
    
    # Load ONNX model
    try:
        session = ort.InferenceSession(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input name: {input_name}")
    print(f"Expected input shape: {input_shape}")
    
    # Load and process image
    try:
        image = Image.open(image_path)
        print(f"Original image size: {image.size}")
        print(f"Original image mode: {image.mode}")
        
        # Resize image to match model input size
        img_size = (224, 224)
        image = image.resize(img_size)
        print(f"Resized image size: {image.size}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("✓ Converted to RGB")
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32)
        print(f"Image array shape before normalization: {image_array.shape}")
        print(f"Image array range before normalization: [{image_array.min():.2f}, {image_array.max():.2f}]")
        
        # POTENTIAL ISSUE 1: Check normalization
        # Different models expect different normalization
        # Option 1: Simple 0-1 normalization (your current approach)
        image_array_v1 = image_array / 255.0
        
        # Option 2: ImageNet normalization (more common for ResNet)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array_v2 = (image_array / 255.0 - mean) / std
        
        print(f"Normalized v1 range: [{image_array_v1.min():.2f}, {image_array_v1.max():.2f}]")
        print(f"Normalized v2 range: [{image_array_v2.min():.2f}, {image_array_v2.max():.2f}]")
        
        # Try both normalization approaches
        for version, norm_array in [("Simple 0-1", image_array_v1), ("ImageNet", image_array_v2)]:
            print(f"\n--- Testing with {version} normalization ---")
            
            # Transpose and add batch dimension
            processed_array = np.transpose(norm_array, (2, 0, 1))
            input_data = np.expand_dims(processed_array, axis=0)
            
            print(f"Final input shape: {input_data.shape}")
            print(f"Final input range: [{input_data.min():.2f}, {input_data.max():.2f}]")
            
            # POTENTIAL ISSUE 2: Check input data type
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
                print("✓ Converted to float32")
            
            # Run inference
            try:
                outputs = session.run(None, {input_name: input_data})
                predictions = outputs[0]
                
                print(f"Raw predictions shape: {predictions.shape}")
                print(f"Raw predictions: {predictions[0]}")
                
                # POTENTIAL ISSUE 3: Check if predictions are all the same
                if len(set(predictions[0])) == 1:
                    print("WARNING: All prediction scores are identical!")
                    print("This suggests the model might not be working correctly")
                
                # Get predicted class
                predicted_class = np.argmax(predictions)
                confidence = predictions[0][predicted_class]
                
                print(f"Predicted class index: {predicted_class}")
                print(f"Confidence: {confidence:.4f}")
                
                # Load labels
                with open(labels_file, 'r') as f:
                    labels = [line.strip() for line in f.readlines()]
                
                print(f"Number of labels: {len(labels)}")
                print(f"Labels: {labels}")
                
                if predicted_class < len(labels):
                    label_name = labels[predicted_class]
                    print(f"Predicted label: {label_name}")
                else:
                    print(f"ERROR: Predicted class {predicted_class} exceeds label count {len(labels)}")
                    
            except Exception as e:
                print(f"ERROR during inference: {e}")
    
    except Exception as e:
        print(f"ERROR processing image: {e}")

# Additional debugging functions
def test_multiple_images(image_paths, model_path, labels_file):
    """Test multiple images to see if outputs differ"""
    print("\n=== TESTING MULTIPLE IMAGES ===")
    results = []
    
    for img_path in image_paths:
        if os.path.exists(img_path):
            print(f"\nTesting: {img_path}")
            # Quick test version
            session = ort.InferenceSession(model_path)
            image = Image.open(img_path)
            image = image.resize((224, 224))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = np.array(image).astype(np.float32) / 255.0
            image_array = np.transpose(image_array, (2, 0, 1))
            input_data = np.expand_dims(image_array, axis=0)
            
            outputs = session.run(None, {session.get_inputs()[0].name: input_data})
            predicted_class = np.argmax(outputs[0])
            confidence = outputs[0][0][predicted_class]
            
            results.append((img_path, predicted_class, confidence))
            print(f"  -> Class: {predicted_class}, Confidence: {confidence:.4f}")
    
    # Check if all results are the same
    if len(set(result[1] for result in results)) == 1:
        print("\nWARNING: All images produced the same prediction!")
        print("This confirms there's likely an issue with the model or preprocessing")
    else:
        print("\n✓ Different images produced different predictions")

# Usage example:
if __name__ == "__main__":
    # Your original paths
    image_path = "/home/nvidia/finalProject/locks/KeyL/test/Locker_lock_H6312R-KD.JPG"
    model_path = "/home/nvidia/finalProject/models/resnet18_v3.onnx"
    labels_file = "/home/nvidia/finalProject/models/labels.txt"
    
    # Run debug
    debug_onnx_classifier(image_path, model_path, labels_file)
    
    # Test multiple images if available
    test_images = [
        "/home/nvidia/finalProject/com1.png",
        "/home/nvidia/finalProject/com2.png",  # Add more test images
        "/home/nvidia/finalProject/com3.png"
    ]
    
    # Uncomment to test multiple images
    # test_multiple_images(test_images, model_path, labels_file)