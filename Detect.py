import cv2
import numpy as np
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt
from scipy import ndimage
import torch

class ChangeDetectionSAM:
    def __init__(self, sam_model_name="facebook/sam-vit-base"):
        
        self.model = SamModel.from_pretrained(sam_model_name)
        self.processor = SamProcessor.from_pretrained(sam_model_name)
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def compute_change_mask(self, img1, img2, threshold=30):
        
        # Convert PIL to numpy if needed
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)
            
        # Convert to grayscale if colored
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1
            
        if len(img2.shape) == 3:
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img2_gray = img2
            
        # Compute absolute difference
        difference = cv2.absdiff(img1_gray, img2_gray)
        
        # Apply threshold to get binary mask
        _, change_mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
        
        return change_mask, difference
    
    def find_significant_contours(self, change_mask, min_area=500, max_contours=10):
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours
        contour_info = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Calculate additional metrics
            perimeter = cv2.arcLength(contour, True)
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            contour_info.append({
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'compactness': compactness,
                'aspect_ratio': aspect_ratio,
                'centroid': (cx, cy),
                'bbox': (x, y, w, h),
                'score': area * compactness  # Combined score for ranking
            })
        
        # Sort by score (area * compactness) in descending order
        contour_info.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top contours
        significant_contours = [info['contour'] for info in contour_info[:max_contours]]
        
        return significant_contours, contour_info[:max_contours]
    
    def contours_to_sam_points(self, contour_info, strategy='centroid'):
        
        input_points = []
        
        for info in contour_info:
            if strategy == 'centroid':
                input_points.append(list(info['centroid']))
            elif strategy == 'bbox_center':
                x, y, w, h = info['bbox']
                center = [x + w//2, y + h//2]
                input_points.append(center)
            elif strategy == 'multiple_points':
                # Add centroid and a few points along the contour
                input_points.append(list(info['centroid']))
                contour = info['contour']
                if len(contour) > 4:
                    # Sample a few points from the contour
                    indices = np.linspace(0, len(contour)-1, min(3, len(contour)), dtype=int)
                    for idx in indices:
                        point = contour[idx][0]  # contour points are in format [[x,y]]
                        input_points.append([int(point[0]), int(point[1])])
        
        return input_points
    
    def segment_with_sam(self, image, input_points):
        
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray(image).convert("RGB")
        
        # Prepare input for SAM
        # SAM expects input_points in format [[[x1, y1], [x2, y2], ...]]
        formatted_points = [input_points] if input_points else None
        
        inputs = self.processor(
            image, 
            input_points=formatted_points, 
            return_tensors="pt"
        ).to(self.device)
        
        # Run SAM
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process masks
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )
        
        scores = outputs.iou_scores.cpu().numpy()
        
        return masks, scores
    
    def process_change_detection(self, img1, img2, threshold=30, min_area=500, 
                               max_contours=5, point_strategy='centroid'):
        
        # Step 1: Compute change mask
        change_mask, difference = self.compute_change_mask(img1, img2, threshold)
        
        # Step 2: Find significant contours
        contours, contour_info = self.find_significant_contours(
            change_mask, min_area, max_contours
        )
        
        if not contours:
            return {
                'change_mask': change_mask,
                'difference': difference,
                'contours': [],
                'sam_masks': None,
                'sam_scores': None,
                'input_points': []
            }
        
        # Step 3: Convert contours to SAM input points
        input_points = self.contours_to_sam_points(contour_info, point_strategy)
        
        # Step 4: Run SAM segmentation on the original image (or difference)
        # You can choose to segment on img1, img2, or the difference image
        masks, scores = self.segment_with_sam(img2, input_points)  # Using img2 as target
        
        return {
            'change_mask': change_mask,
            'difference': difference,
            'contours': contours,
            'contour_info': contour_info,
            'input_points': input_points,
            'sam_masks': masks,
            'sam_scores': scores
        }
    
    def process_and_save(self, img1, img2, results, save_path='result'):
        """
        Visualize the complete pipeline results
        """        
        # Contours on original image
        img_with_contours = img2.copy() if isinstance(img2, np.ndarray) else np.array(img2)
        if len(img_with_contours.shape) == 2:
            img_with_contours = cv2.cvtColor(img_with_contours, cv2.COLOR_GRAY2RGB)
        
        cv2.drawContours(img_with_contours, results['contours'], -1, (0, 255, 0), 2)
        
        # Draw input points
        for point in results['input_points']:
            cv2.circle(img_with_contours, tuple(point), 5, (255, 0, 0), -1)
        
        cv2.imwrite(f"./Results/{save_path}_segmented.jpg", img_with_contours)
        
    

if __name__ == "__main__":
    
    
    detector = ChangeDetectionSAM()
    
    path1 = './TestImages/ultrasonicsensor1.jpg'
    path2 = './TestImages/ultrasonicsensor2.jpg'
    
    # Load your images (replace with your actual images)
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    # candle\Data\Images\Normal\0000.JPG
    print(f"Image shapes: {img1.shape}, {img2.shape}")
    # Process the images
    results = detector.process_change_detection(
        img1, img2,
        threshold=70,
        min_area=100,
        max_contours=5,
        point_strategy='centroid'
    )
    
    print(f"Results: {results.keys()}")
    # Visualize results
    detector.process_and_save(img1, img2, results, save_path=path2[path2.find("./TestImages/")+len("./TestImages/"):path2.find(".jpg")-1])
    
    # Print information about detected changes
    print(f"Number of significant contours found: {len(results['contours'])}")
    print(f"SAM input points: {results['input_points']}")
    if results['sam_scores'] is not None:
        print(f"SAM scores: {results['sam_scores'][0]}")