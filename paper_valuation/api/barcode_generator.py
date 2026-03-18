import qrcode
from PIL import Image
from io import BytesIO

def generate_qr_code(barcode_id):
    """
    Generate QR Code (much more reliable than CODE128)
    """
    try:
        # Create QR code instance
        qr = qrcode.QRCode(
            version=1,  # Size (1-40, 1 is smallest)
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction (30%)
            box_size=20,  # Size of each box in pixels
            border=4,  # Border size (minimum is 4)
        )
        
        # Add data
        qr.add_data(barcode_id)
        qr.make(fit=True)
        
        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to PIL Image
        if not isinstance(img, Image.Image):
            img = img.convert('RGB')
        
        print(f"   Generated QR code size: {img.size}")
        
        return img
        
    except Exception as e:
        print(f"❌ Error generating QR code for {barcode_id}: {str(e)}")
        raise