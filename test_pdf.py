from pdf2image import convert_from_path

# 1. Define your Poppler path (using 'r' to handle backslashes)
my_poppler_path = r'C:\poppler\poppler-25.12.0\Library\bin'

pdf_path = r'E:\full stack\projects\paper-valuation\generated_pdfs\facing_sheets_batch_14.pdf'

# 2. Pass the poppler_path into the function
images = convert_from_path(
    pdf_path, 
    first_page=1, 
    last_page=1, 
    dpi=600, 
    poppler_path=my_poppler_path  # <--- Add this line
)

output_path = r'E:\machine learning\project\paper-valuation\test\imagee_600dpi.png'
images[0].save(output_path, 'PNG')

print(f"✅ Saved at 600 DPI: {output_path}")