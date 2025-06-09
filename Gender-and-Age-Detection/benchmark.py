# Batch Gender and Age Detection program adapted from Mahesh Sawant's original code
import cv2
import math
import argparse
import os
import csv
import glob
from pathlib import Path

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    confidences = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            confidences.append(confidence)
    
    return frameOpencvDnn, faceBoxes, confidences

def process_single_image(image_path, faceNet, ageNet, genderNet, MODEL_MEAN_VALUES, ageList, genderList, padding=20):
    """Process a single image and return results"""
    try:
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        
        # Detect faces
        resultImg, faceBoxes, confidences = highlightFace(faceNet, frame)
        
        if not faceBoxes:
            return {
                'filename': os.path.basename(image_path),
                'face_detected': False,
                'confidence': 0,
                'gender': 'N/A',
                'age': 'N/A',
                'error': None
            }
        
        # Find the face with highest confidence
        max_conf_idx = confidences.index(max(confidences))
        best_faceBox = faceBoxes[max_conf_idx]
        best_confidence = confidences[max_conf_idx]
        
        # Extract the face region
        face = frame[max(0, best_faceBox[1] - padding):
                    min(best_faceBox[3] + padding, frame.shape[0] - 1),
                    max(0, best_faceBox[0] - padding):
                    min(best_faceBox[2] + padding, frame.shape[1] - 1)]
        
        # Predict gender
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        return {
            'filename': os.path.basename(image_path),
            'face_detected': True,
            'confidence': float(best_confidence),
            'gender': gender,
            'age': age,
            'total_faces': len(faceBoxes),
            'error': None
        }
        
    except Exception as e:
        return {
            'filename': os.path.basename(image_path),
            'face_detected': False,
            'confidence': 0,
            'gender': 'N/A',
            'age': 'N/A',
            'total_faces': 0,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, help='Directory containing images to process')
    parser.add_argument('--output', default='results.csv', help='Output CSV file name')
    parser.add_argument('--conf_threshold', type=float, default=0.7, help='Confidence threshold for face detection')
    args = parser.parse_args()
    
    # Model files
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    
    # Load networks
    print("Loading networks...")
    try:
        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)
        print("Networks loaded successfully!")
    except Exception as e:
        print(f"Error loading networks: {e}")
        return
    
    # Get all image files from directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.directory, extension.lower())))
        image_files.extend(glob.glob(os.path.join(args.directory, extension.upper())))
    
    if not image_files:
        print(f"No image files found in directory: {args.directory}")
        return
    
    print(f"Found {len(image_files)} images to process...")
    
    # Process all images
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
        result = process_single_image(
            image_path, faceNet, ageNet, genderNet, 
            MODEL_MEAN_VALUES, ageList, genderList
        )
        if result:
            results.append(result)
    
    # Save results to CSV
    if results:
        with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'face_detected', 'confidence', 'gender', 'age', 'total_faces', 'error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"\nResults saved to: {args.output}")
        
        # Print summary
        successful = sum(1 for r in results if r['face_detected'])
        failed = len(results) - successful
        print(f"Summary: {successful} images processed successfully, {failed} failed")
        
        if successful > 0:
            print("\nSample results:")
            for result in results[:5]:  # Show first 5 results
                if result['face_detected']:
                    print(f"  {result['filename']}: {result['gender']}, {result['age']}, confidence: {result['confidence']:.3f}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()