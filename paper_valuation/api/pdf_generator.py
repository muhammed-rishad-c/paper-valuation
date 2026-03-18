from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from io import BytesIO
import os
from .barcode_generator import generate_qr_code

import qrcode
from io import BytesIO
from PIL import Image

def generate_qr_code(barcode_id):
    """
    Generate QR code for barcode ID
    Returns PIL Image object
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
        box_size=20,
        border=4,
    )
    
    # Add barcode ID data
    qr.add_data(barcode_id)
    qr.make(fit=True)
    
    # Create image
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    return qr_img

def generate_facing_sheet_pdf(mappings, exam_details, output_path):
    """
    Generate professional KTU facing sheet with QR code
    """
    try:
        print(f"📄 Generating PDF for {len(mappings)} students...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4
        
        for idx, mapping in enumerate(mappings):
            print(f"   Page {idx + 1}/{len(mappings)}: {mapping['barcode_id']} - {mapping['student_name']}")
            
            draw_ktu_professional_sheet(c, width, height, mapping, exam_details)
            
            if idx < len(mappings) - 1:
                c.showPage()
        
        c.save()
        print(f"✅ PDF generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def draw_ktu_professional_sheet(c, width, height, mapping, exam_details):
    """
    Professional KTU facing sheet with perfect alignment
    """
    barcode_id = mapping['barcode_id']
    
    # Page margins
    left_margin = 20*mm
    right_margin = 20*mm
    top_margin = 15*mm
    
    # Colors
    border_color = HexColor('#000000')
    text_color = HexColor('#000000')
    
    c.setStrokeColor(border_color)
    c.setFillColor(text_color)
    
    # ============================================
    # TOP SECTION - QUESTION PAPER CODE & QR CODE
    # ============================================
    
    current_y = height - top_margin - 5*mm
    
    # Left Box - Question Paper Code (empty for exam officer)
    qpc_box_width = 65*mm
    qpc_box_height = 28*mm
    
    c.setLineWidth(1.2)
    c.rect(left_margin, current_y - qpc_box_height, qpc_box_width, qpc_box_height, stroke=1, fill=0)
    
    # Label below box
    c.setFont("Helvetica", 9)
    c.drawString(left_margin + 2*mm, current_y - qpc_box_height - 5*mm, "Question Paper Code")
    
    # Right Box - QR CODE & Alpha Numeric
    qr_section_x = width - right_margin - 72*mm
    qr_box_width = 72*mm
    qr_box_height = 38*mm
    
    c.rect(qr_section_x, current_y - qr_box_height, qr_box_width, qr_box_height, stroke=1, fill=0)
    
    # Generate and place QR CODE
    try:
        qr_img = generate_qr_code(barcode_id)
        img_buffer = BytesIO()
        qr_img.save(img_buffer, format='PNG', dpi=(600, 600))
        img_buffer.seek(0)
        
        # QR code size and position (centered in box)
        qr_size = 32*mm
        qr_x = qr_section_x + (qr_box_width - qr_size) / 2
        qr_y = current_y - 35*mm
        
        img_reader = ImageReader(img_buffer)
        c.drawImage(
            img_reader,
            qr_x,
            qr_y,
            width=qr_size,
            height=qr_size,
            preserveAspectRatio=True,
            mask='auto'
        )
    except Exception as e:
        print(f"⚠️ QR code generation failed: {str(e)}")
    
    # "Alpha Numeric Code" label
    c.setFont("Helvetica", 8)
    alpha_label_y = current_y - qr_box_height - 5*mm
    c.drawString(qr_section_x + 2*mm, alpha_label_y, "Alpha Numeric Code")
    
    # Alpha numeric boxes with characters
    box_start_y = alpha_label_y - 8*mm
    char_box_size = 4.8*mm
    char_box_height = 6*mm
    
    # Calculate to center the boxes
    total_boxes_width = len(barcode_id) * char_box_size
    box_start_x = qr_section_x + (qr_box_width - total_boxes_width) / 2
    
    c.setLineWidth(0.8)
    c.setFont("Helvetica-Bold", 9)
    
    for i, char in enumerate(barcode_id):
        box_x = box_start_x + (i * char_box_size)
        c.rect(box_x, box_start_y, char_box_size, char_box_height, stroke=1, fill=0)
        # Center character in box
        c.drawCentredString(box_x + char_box_size/2, box_start_y + 1.5*mm, char)
    
    # ============================================
    # UNIVERSITY HEADER
    # ============================================
    
    header_y = current_y - qr_box_height - 20*mm
    
    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(width/2, header_y, "APJ ABDUL KALAM TECHNOLOGICAL UNIVERSITY")
    
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(width/2, header_y - 8*mm, "ANSWER BOOKLET")
    
    # Horizontal line
    c.setLineWidth(1.5)
    line_y = header_y - 13*mm
    c.line(left_margin, line_y, width - right_margin, line_y)
    
    # ============================================
    # PACKET INFORMATION
    # ============================================
    
    packet_y = line_y - 10*mm
    
    c.setLineWidth(1)
    c.setFont("Helvetica", 10)
    
    # Packet Code
    c.drawString(left_margin + 5*mm, packet_y, "Packet Code :")
    c.rect(left_margin + 35*mm, packet_y - 3*mm, 60*mm, 7*mm, stroke=1, fill=0)
    
    # Packet Serial No
    serial_x = width - right_margin - 110*mm
    c.drawString(serial_x, packet_y, "Packet Serial No. :")
    c.rect(serial_x + 45*mm, packet_y - 3*mm, 60*mm, 7*mm, stroke=1, fill=0)
    
    # Horizontal line
    c.setLineWidth(1)
    line2_y = packet_y - 10*mm
    c.line(left_margin, line2_y, width - right_margin, line2_y)
    
    # ============================================
    # INSTRUCTIONS TO CANDIDATES
    # ============================================
    
    inst_start_y = line2_y - 8*mm
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left_margin + 3*mm, inst_start_y, "Instructions to Candidates:")
    
    # Instructions text
    instructions = [
        "1. Do not write your Name or Register Number anywhere in this Answer Booklet.",
        "2. Peel the Barcodes corresponding to your Register Number and paste one on the Attendance Sheet and one",
        "    on the last page.",
        "3. Affix your signature on the space provided in the Attendance Sheet.",
        "4. Do not tamper or write on the barcodes as it will mutilate the code. The action will lead to debarring from",
        "5. examinations and withholding the results.",
        "6. Please confine the answers within the space provided in the Answer Booklet. No additional sheets will be",
        "    supplied.",
        "7. DO NOT WRITE ON THE REVERSE OF THIS PAGE.",
        "8. All question numbers and labels must be written clearly inside the left-hand margin column only. Keep",
        "    this area clean and free of miscellaneous notes.",
        "9. If an answer continues onto a new sheet, you must re-write the question number at the top of the left",
        "    margin on the new page.",
        "10. Use only ballpoint pens (black or blue ink).",
        "11. Ensure all handwriting is legible and stays within the designated writing areas.",
    ]
    
    c.setFont("Helvetica", 8.5)
    inst_y = inst_start_y - 6*mm
    line_spacing = 4*mm
    
    for instruction in instructions:
        c.drawString(left_margin + 3*mm, inst_y, instruction)
        inst_y -= line_spacing
    
    # Note section
    note_y = inst_y - 4*mm
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(left_margin + 3*mm, note_y, "Note:")
    
    c.setFont("Helvetica", 8.5)
    note_text = "Failure to follow these formatting rules may result in delays in grading or loss of marks due to illegibility."
    c.drawString(left_margin + 15*mm, note_y, note_text)
    
    # Horizontal line
    line3_y = note_y - 7*mm
    c.setLineWidth(1)
    c.line(left_margin, line3_y, width - right_margin, line3_y)
    
    # ============================================
    # EXAM DETAILS SECTION
    # ============================================
    
    details_y = line3_y - 10*mm
    
    c.setFont("Helvetica", 10)
    exam_line = "Name of Exam:………….....SEMESTER………………DEGREE EXAMINATION…………………….20………"
    c.drawString(left_margin + 3*mm, details_y, exam_line)
    
    # Course Code
    details_y -= 13*mm
    c.setFont("Helvetica", 10)
    c.drawString(left_margin + 25*mm, details_y + 3*mm, "Course Code:")
    
    draw_aligned_boxes(c, left_margin + 58*mm, details_y, 18, 5*mm, 6.5*mm)
    
    # Branch/Stream
    details_y -= 11*mm
    c.drawString(left_margin + 18*mm, details_y + 3*mm, "Branch/Stream:")
    
    draw_aligned_boxes(c, left_margin + 58*mm, details_y, 18, 5*mm, 6.5*mm)
    
    # Course (Subject) Name - Two rows
    details_y -= 13*mm
    c.drawString(left_margin + 3*mm, details_y + 10*mm, "Course (Subject) Name:")
    
    # First row
    draw_aligned_boxes(c, left_margin + 58*mm, details_y + 7*mm, 22, 5*mm, 6.5*mm)
    
    # Second row
    draw_aligned_boxes(c, left_margin + 58*mm, details_y, 22, 5*mm, 6.5*mm)

def draw_aligned_boxes(c, start_x, start_y, count, box_width, box_height):
    """
    Draw a row of perfectly aligned empty boxes
    """
    c.setLineWidth(0.8)
    for i in range(count):
        box_x = start_x + (i * box_width)
        c.rect(box_x, start_y, box_width, box_height, stroke=1, fill=0)