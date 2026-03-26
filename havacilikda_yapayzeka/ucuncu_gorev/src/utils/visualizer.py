import cv2
import numpy as np

def draw_matches(image, results, output_path):
    """Bulunan eşleşmeleri kare üzerine çizer."""
    viz = image.copy()
    for res in results:
        if res.found and res.bbox:
            x1, y1, x2, y2 = map(int, res.bbox)
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(viz, f"{res.reference_id} ({res.confidence:.2f})", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, viz)
