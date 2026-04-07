import json
import random
import math
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
        Simulate fruit detection with more realistic bounding boxes
        """
        try:
            # Simulate image processing
            img_width, img_height = 640, 480  # Assume standard image size
            
            # More realistic detection based on image content simulation
            detections = []
            
            # Simulate analyzing image content (in real app, this would use actual ML)
            # For demo, create more realistic fruit arrangements
            fruit_scenarios = [
                # Single fruit scenarios
                [{'fruits': ['banana'], 'count': 1, 'spread': 'tight'}],
                [{'fruits': ['apple'], 'count': 1, 'spread': 'tight'}],
                [{'fruits': ['orange'], 'count': 1, 'spread': 'tight'}],
                [{'fruits': ['grape'], 'count': 1, 'spread': 'tight'}],
                # Multiple fruits scenarios
                [{'fruits': ['apple', 'banana'], 'count': 2, 'spread': 'medium'}],
                [{'fruits': ['orange', 'grape'], 'count': 2, 'spread': 'medium'}],
                [{'fruits': ['strawberry', 'mango'], 'count': 2, 'spread': 'medium'}],
                # Mixed bowl scenario
                [{'fruits': ['apple', 'orange', 'banana'], 'count': 3, 'spread': 'bowl'}],
                [{'fruits': ['grape', 'strawberry'], 'count': 2, 'spread': 'cluster'}],
            ]
            
            # Select a realistic scenario
            scenario = random.choice(fruit_scenarios)
            
            # Calculate positions to avoid overlap and be realistic
            positions = self._calculate_realistic_positions(
                scenario['fruits'], 
                img_width, 
                img_height, 
                scenario['spread']
            )
            
            for i, fruit_key in enumerate(scenario['fruits']):
                fruit = self.fruit_classes[fruit_key]
                pos = positions[i]
                
                # Generate realistic confidence based on position and fruit type
                base_confidence = random.uniform(*fruit.confidence_range)
                
                # Adjust confidence based on position (center = higher confidence)
                center_x, center_y = img_width // 2, img_height // 2
                distance_from_center = ((pos['x'] - center_x)**2 + (pos['y'] - center_y)**2)**0.5
                max_distance = (center_x**2 + center_y**2)**0.5
                position_factor = 1.0 - (distance_from_center / max_distance) * 0.2
                
                confidence = min(0.99, base_confidence * position_factor)
                
                detection = {
                    'class': fruit.name,
                    'class_id': list(self.fruit_classes.keys()).index(fruit_key),
                    'confidence': round(confidence, 3),
                    'bbox': {
                        'x_min': pos['x'],
                        'y_min': pos['y'],
                        'x_max': pos['x'] + pos['width'],
                        'y_max': pos['y'] + pos['height'],
                        'width': pos['width'],
                        'height': pos['height']
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
    
    def _calculate_realistic_positions(self, fruits: List[str], img_width: int, img_height: int, spread_type: str) -> List[Dict]:
        """Calculate realistic fruit positions to avoid overlap"""
        positions = []
        
        if spread_type == 'tight':
            # Single fruit, centered
            fruit_key = fruits[0]
            fruit = self.fruit_classes[fruit_key]
            x = (img_width - fruit.typical_size[0]) // 2
            y = (img_height - fruit.typical_size[1]) // 2
            positions.append({
                'x': x + random.randint(-10, 10),
                'y': y + random.randint(-10, 10),
                'width': fruit.typical_size[0],
                'height': fruit.typical_size[1]
            })
            
        elif spread_type == 'medium':
            # Two fruits, side by side or slightly overlapping
            for i, fruit_key in enumerate(fruits):
                fruit = self.fruit_classes[fruit_key]
                if i == 0:
                    x = img_width // 3
                    y = img_height // 2
                else:
                    x = 2 * img_width // 3
                    y = img_height // 2
                
                positions.append({
                    'x': x - fruit.typical_size[0] // 2,
                    'y': y - fruit.typical_size[1] // 2,
                    'width': fruit.typical_size[0],
                    'height': fruit.typical_size[1]
                })
                
        elif spread_type == 'bowl':
            # Multiple fruits in a circular arrangement
            center_x, center_y = img_width // 2, img_height // 2
            radius = 80
            angle_step = 360 // len(fruits)
            
            for i, fruit_key in enumerate(fruits):
                fruit = self.fruit_classes[fruit_key]
                angle = i * angle_step
                x = center_x + radius * math.cos(math.radians(angle)) - fruit.typical_size[0] // 2
                y = center_y + radius * math.sin(math.radians(angle)) - fruit.typical_size[1] // 2
                
                positions.append({
                    'x': int(x),
                    'y': int(y),
                    'width': fruit.typical_size[0],
                    'height': fruit.typical_size[1]
                })
                
        elif spread_type == 'cluster':
            # Fruits clustered together
            base_x, base_y = img_width // 2 - 50, img_height // 2 - 30
            
            for i, fruit_key in enumerate(fruits):
                fruit = self.fruit_classes[fruit_key]
                offset_x = (i % 2) * 60
                offset_y = (i // 2) * 40
                
                positions.append({
                    'x': base_x + offset_x,
                    'y': base_y + offset_y,
                    'width': fruit.typical_size[0],
                    'height': fruit.typical_size[1]
                })
        
        return positions

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
