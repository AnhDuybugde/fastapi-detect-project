import json
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64

@dataclass
class FruitClass:
    name: str
    color: str
    typical_size: Tuple[int, int]  # (width, height)
    confidence_range: Tuple[float, float]

class FruitDetector:
    def __init__(self):
        # Define fruit classes with realistic properties
        self.fruit_classes = {
            'banana': FruitClass('banana', '#FFE135', (80, 200), (0.85, 0.98)),
            'apple': FruitClass('apple', '#FF0000', (70, 70), (0.80, 0.95)),
            'orange': FruitClass('orange', '#FFA500', (75, 75), (0.82, 0.96)),
            'grape': FruitClass('grape', '#800080', (40, 60), (0.75, 0.92)),
            'watermelon': FruitClass('watermelon', '#00FF00', (200, 150), (0.88, 0.99)),
            'strawberry': FruitClass('strawberry', '#FF1493', (30, 35), (0.78, 0.94)),
            'mango': FruitClass('mango', '#FFD700', (90, 120), (0.83, 0.97)),
            'pineapple': FruitClass('pineapple', '#8B4513', (100, 180), (0.80, 0.95))
        }
        
        # Common fruit combinations in images
        self.fruit_combinations = [
            ['banana', 'apple', 'orange'],
            ['grape', 'strawberry'],
            ['watermelon', 'mango'],
            ['apple', 'banana', 'grape'],
            ['orange', 'pineapple'],
            ['strawberry', 'mango', 'banana']
        ]

    def detect_fruits(self, image_data: bytes) -> List[Dict]:
        """
        Simulate fruit detection with realistic bounding boxes
        """
        # Simulate image processing
        try:
            # In real implementation, this would use YOLO/other ML model
            # For demo, we'll generate realistic mock detections
            
            # Randomly select fruit combination for this "detection"
            selected_fruits = random.choice(self.fruit_combinations)
            
            detections = []
            img_width, img_height = 640, 480  # Assume standard image size
            
            # Generate bounding boxes for selected fruits
            for i, fruit_key in enumerate(selected_fruits):
                fruit = self.fruit_classes[fruit_key]
                
                # Generate realistic position and size
                x_min = random.randint(50, img_width - fruit.typical_size[0] - 50)
                y_min = random.randint(50, img_height - fruit.typical_size[1] - 50)
                x_max = x_min + fruit.typical_size[0] + random.randint(-20, 20)
                y_max = y_min + fruit.typical_size[1] + random.randint(-20, 20)
                
                # Generate realistic confidence score
                confidence = random.uniform(*fruit.confidence_range)
                
                detection = {
                    'class': fruit.name,
                    'class_id': list(self.fruit_classes.keys()).index(fruit_key),
                    'confidence': round(confidence, 3),
                    'bbox': {
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'width': x_max - x_min,
                        'height': y_max - y_min
                    },
                    'color': fruit.color
                }
                detections.append(detection)
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def draw_detections(self, image_data: bytes, detections: List[Dict]) -> str:
        """
        Draw bounding boxes on image and return base64 encoded result
        """
        try:
            # Create a blank image with fruits drawn
            img = Image.new('RGB', (640, 480), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw simple representations of fruits
            for detection in detections:
                bbox = detection['bbox']
                color = detection['color']
                
                # Draw rectangle for fruit
                draw.rectangle([
                    bbox['x_min'], bbox['y_min'],
                    bbox['x_max'], bbox['y_max']
                ], outline=color, width=3)
                
                # Draw label
                label = f"{detection['class']} {detection['confidence']:.2f}"
                draw.rectangle([
                    bbox['x_min'], bbox['y_min'] - 25,
                    bbox['x_min'] + len(label) * 8, bbox['y_min']
                ], fill=color, outline=color)
                
                # Add text (simplified - in real implementation would use font)
                draw.text((bbox['x_min'] + 2, bbox['y_min'] - 22), 
                         label, fill='white')
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            print(f"Drawing error: {e}")
            return ""

    def get_fruit_info(self) -> Dict:
        """Get information about all detectable fruits"""
        return {
            'classes': list(self.fruit_classes.keys()),
            'total_classes': len(self.fruit_classes),
            'model_info': {
                'name': 'FruitDetector v1.0',
                'type': 'YOLO-based simulation',
                'input_size': '640x640',
                'confidence_threshold': 0.5
            }
        }

# Global detector instance
fruit_detector = FruitDetector()

def detect_fruits_in_image(image_data: bytes) -> Dict:
    """
    Main function to detect fruits in image
    """
    detections = fruit_detector.detect_fruits(image_data)
    
    return {
        'detections': detections,
        'total_detections': len(detections),
        'detected_classes': list(set(d['class'] for d in detections)),
        'model_info': fruit_detector.get_fruit_info()
    }

def get_available_fruits() -> Dict:
    """Get list of fruits that can be detected"""
    return fruit_detector.get_fruit_info()
