import cv2
import numpy as np
import re
from pyzbar.pyzbar import decode
import pytesseract
import os

# Configure tesseract path
tesseract_exe = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_exe):
    pytesseract.pytesseract.tesseract_cmd = tesseract_exe
    print(f"✅ Tesseract configured: {tesseract_exe}")

def scan_barcode_from_image(image_path):
    """
    Scan QR code or OCR text from facing sheet
    Priority: QR Code → OCR large text → Manual entry
    """
    try:
        print(f"\n{'='*60}")
        print(f"📸 Processing facing sheet: {image_path}")
        print(f"{'='*60}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {
                'success': False,
                'barcode_id': None,
                'error': 'Could not read image file'
            }
        
        print(f"   📐 Image size: {img.shape[1]}x{img.shape[0]} pixels")
        
        # STEP 1: Try QR Code scanning (PRIMARY METHOD)
        print("\n   🔍 METHOD 1: QR Code scanning...")
        barcode_id = scan_qr_code(img)
        
        if barcode_id:
            print(f"\n   ✅ SUCCESS! Barcode ID: {barcode_id}")
            print(f"   📍 Detection method: QR Code")
            return {
                'success': True,
                'barcode_id': barcode_id,
                'barcode_type': 'QR',
                'method': 'qr_code',
                'confidence': 'high'
            }
        
        # STEP 2: Try OCR on alpha numeric boxes (SECONDARY)
        print("\n   🔍 METHOD 2: OCR on alpha numeric boxes...")
        barcode_id = ocr_alpha_numeric_boxes(img)
        
        if barcode_id:
            print(f"\n   ✅ SUCCESS! Barcode ID: {barcode_id}")
            print(f"   📍 Detection method: OCR (Alpha Numeric)")
            return {
                'success': True,
                'barcode_id': barcode_id,
                'barcode_type': 'OCR',
                'method': 'ocr_alpha_numeric',
                'confidence': 'medium'
            }
        
        # STEP 3: Try full image OCR (TERTIARY)
        print("\n   🔍 METHOD 3: Full image OCR scan...")
        barcode_id = ocr_full_image(img)
        
        if barcode_id:
            print(f"\n   ✅ SUCCESS! Barcode ID: {barcode_id}")
            print(f"   📍 Detection method: OCR (Full Image)")
            return {
                'success': True,
                'barcode_id': barcode_id,
                'barcode_type': 'OCR',
                'method': 'ocr_full_image',
                'confidence': 'low'
            }
        
        # STEP 4: Failed - needs manual entry
        print("\n   ❌ Auto-detection failed")
        print("   💡 Suggestion: Manual entry required")
        
        return {
            'success': False,
            'barcode_id': None,
            'barcode_type': None,
            'method': 'failed',
            'error': 'Could not detect barcode ID automatically',
            'suggestion': 'Please enter barcode ID manually'
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'barcode_id': None,
            'error': str(e)
        }

def scan_qr_code(img):
    """
    Scan QR code using pyzbar
    """
    try:
        # Try multiple preprocessing techniques
        techniques = [
            ('original', img),
            ('grayscale', cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        ]
        
        # Add upscaled versions
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        techniques.append(('upscale_2x', cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)))
        
        for method_name, processed_img in techniques:
            print(f"      → Trying {method_name}...")
            
            # Decode QR codes
            decoded_objects = decode(processed_img)
            
            for obj in decoded_objects:
                if obj.type == 'QRCODE':
                    barcode_data = obj.data.decode('utf-8')
                    print(f"         Raw QR data: {barcode_data}")
                    
                    # Validate and clean
                    barcode_id = validate_barcode_format(barcode_data)
                    if barcode_id:
                        print(f"         ✅ Valid QR code: {barcode_id}")
                        return barcode_id
        
        print(f"      ❌ No QR code detected")
        return None
        
    except Exception as e:
        print(f"      ⚠️ QR scan error: {e}")
        return None

def ocr_alpha_numeric_boxes(img):
    """
    OCR the alpha numeric code boxes (individual character boxes)
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Focus on right side where alpha numeric boxes are
        regions_to_try = [
            (0.05, 0.25, 0.50, 1.0, "top-right area"),
            (0.10, 0.20, 0.55, 0.95, "focused boxes"),
            (0.08, 0.22, 0.52, 0.98, "wide scan"),
        ]
        
        for y1, y2, x1, x2, desc in regions_to_try:
            print(f"      → Scanning {desc}...")
            
            y_start = int(height * y1)
            y_end = int(height * y2)
            x_start = int(width * x1)
            x_end = int(width * x2)
            
            box_region = gray[y_start:y_end, x_start:x_end]
            
            if box_region.size == 0:
                continue
            
            # Preprocess
            box_region = cv2.resize(box_region, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            box_region = cv2.fastNlMeansDenoising(box_region, h=10)
            box_region = cv2.convertScaleAbs(box_region, alpha=1.5, beta=10)
            _, box_region = cv2.threshold(box_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR with whitelist
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=EXAM0123456789- '
            text = pytesseract.image_to_string(box_region, config=custom_config)
            
            if text.strip():
                print(f"         Raw OCR: '{text.strip()}'")
            
            barcode_id = extract_barcode_id(text)
            if barcode_id:
                print(f"         ✅ Extracted: {barcode_id}")
                return barcode_id
        
        print(f"      ❌ No valid ID in alpha numeric boxes")
        return None
        
    except Exception as e:
        print(f"      ⚠️ OCR boxes error: {e}")
        return None

def ocr_full_image(img):
    """
    Search entire image for barcode ID pattern
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Try different PSM modes
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 3',
            r'--oem 3 --psm 11',
        ]
        
        for config in configs:
            text = pytesseract.image_to_string(gray, config=config)
            barcode_id = extract_barcode_id(text)
            if barcode_id:
                return barcode_id
        
        print(f"      ❌ No valid ID in full image")
        return None
        
    except Exception as e:
        print(f"      ⚠️ Full OCR error: {e}")
        return None

def extract_barcode_id(text):
    """
    Extract and validate barcode ID from OCR text
    """
    if not text:
        return None
    
    text_clean = text.upper().replace('\n', ' ').replace('\r', ' ')
    
    # Pattern 1: EXAM2024-00019 (full format)
    match = re.search(r'EXAM\s*\d{4}\s*-?\s*\d{5}', text_clean)
    if match:
        barcode_id = match.group(0)
        barcode_id = re.sub(r'\s+', '', barcode_id)
        if '-' not in barcode_id:
            barcode_id = re.sub(r'(EXAM\d{4})(\d{5})', r'\1-\2', barcode_id)
        return validate_barcode_format(barcode_id)
    
    # Pattern 2: Separated characters (E X A M 2 0 2 4 - 0 0 0 1 9)
    text_no_spaces = text_clean.replace(' ', '')
    match = re.search(r'EXAM\d{9}', text_no_spaces)
    if match:
        barcode_id = match.group(0)
        barcode_id = re.sub(r'(EXAM\d{4})(\d{5})', r'\1-\2', barcode_id)
        return validate_barcode_format(barcode_id)
    
    # Pattern 3: Just numbers (2024-00019)
    match = re.search(r'\d{4}\s*-\s*\d{5}', text_clean)
    if match:
        number_part = match.group(0).replace(' ', '')
        barcode_id = f"EXAM{number_part}"
        return validate_barcode_format(barcode_id)
    
    return None

def validate_barcode_format(barcode_id):
    """
    Validate barcode ID format: EXAM####-#####
    """
    if not barcode_id:
        return None
    
    # Clean up
    barcode_id = barcode_id.strip().upper()
    
    # Check format
    pattern = r'^EXAM\d{4}-\d{5}$'
    if re.match(pattern, barcode_id):
        return barcode_id
    
    return None