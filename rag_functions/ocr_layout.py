import fitz
from PIL import Image
Image.LINEAR = Image.BILINEAR
import pytesseract

def extract_text_and_layout_simple(pdf_path):
    """
    Simplified version that uses PyMuPDF's built-in text extraction
    and OCR as fallback for images/scanned documents
    """
    doc = fitz.open(pdf_path)
    all_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Try to extract text directly first (faster for text-based PDFs)
        text = page.get_text()
        
        if text.strip():  # If we got text, use it
            all_text += f"\n--- Page {page_num + 1} ---\n"
            all_text += text
        else:  # If no text, use OCR on the page image
            all_text += f"\n--- Page {page_num + 1} (OCR) ---\n"
            
            # Convert page to image
            pix = page.get_pixmap(dpi=300)
            mode = "RGB" if pix.alpha == 0 else "RGBA"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            
            # Use OCR to extract text
            ocr_text = pytesseract.image_to_string(img)
            all_text += ocr_text
    
    doc.close()
    return all_text

def extract_text_with_layoutparser(pdf_path):
    """
    Original version using layoutparser (requires more dependencies)
    """
    import layoutparser as lp
    
    doc = fitz.open(pdf_path)
    all_text = ""
    
    # Initialize the layout detection model
    model = lp.Detectron2LayoutModel(
        'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )
    
    for page in doc:
        # Convert PDF page to image
        pix = page.get_pixmap(dpi=300)
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        
        # Detect layout elements
        layout = model.detect(img)
        
        # Filter for text blocks
        text_blocks = [b for b in layout if b.type in ['Text', 'Title']]
        
        # Extract text from each block
        for block in text_blocks:
            x1, y1, x2, y2 = map(int, block.coordinates)
            seg = img.crop((x1, y1, x2, y2))
            text = pytesseract.image_to_string(seg)
            all_text += "\n" + text
    
    doc.close()
    return all_text

def extract_tables_pymupdf(pdf_path):
      """Extract tables using PyMuPDF's built-in table detection"""
      doc = fitz.open(pdf_path)
      tables = []

      for page_num in range(len(doc)):
          page = doc[page_num]
          print(f"Checking page {page_num + 1} for tables...")
          
          # Try different table detection strategies
          table_finder = page.find_tables(strategy="lines_strict")
          page_tables = list(table_finder)
          if not page_tables:
              table_finder = page.find_tables(strategy="lines")
              page_tables = list(table_finder)
          if not page_tables:
              table_finder = page.find_tables()
              page_tables = list(table_finder)
          
          print(f"Found {len(page_tables)} tables on page {page_num + 1}")

          for i, table in enumerate(page_tables):
              try:
                  table_data = table.extract()
                  tables.append({
                      'page': page_num + 1,
                      'table_index': i,
                      'data': table_data,
                      'bbox': table.bbox
                  })
                  print(f"Successfully extracted table {i+1} with {len(table_data)} rows")
              except Exception as e:
                  print(f"Error extracting table {i+1}: {e}")

      doc.close()
      print(f"Total tables extracted: {len(tables)}")
      
      # If no tables found, try text-based detection as fallback
      if not tables:
          print("No tables found with PyMuPDF detection, trying text-based approach...")
          tables = extract_tables_from_text(pdf_path)
      
      return tables

def extract_tables_from_text(pdf_path):
    """Reconstruct tables from line-by-line text extraction"""
    doc = fitz.open(pdf_path)
    tables = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for table headers (lines starting with "Table")
            if line.startswith('Table ') and ':' in line:
                print(f"\nFound table header: {line}")
                table_data = []
                i += 1
                
                # Try to reconstruct the table structure
                # Look for the next few lines that might be table content
                table_lines = []
                j = i
                while j < len(lines) and j < i + 50:  # Look ahead max 50 lines
                    current_line = lines[j]
                    
                    # Stop if we hit another table or section
                    if (current_line.startswith('Table ') or 
                        current_line.startswith('Note.') or
                        current_line.startswith('Discussion') or
                        len(current_line) > 100):  # Very long lines are likely prose
                        break
                    
                    table_lines.append(current_line)
                    j += 1
                
                # Now try to group these lines into rows and columns
                # Look for patterns that suggest column headers and data
                potential_headers = []
                potential_data = []
                
                for line in table_lines[:20]:  # First 20 lines after table header
                    words = line.split()
                    
                    # Potential column headers (non-numeric, reasonable length)
                    if (len(words) <= 4 and 
                        not any(char.isdigit() for char in line) and 
                        len(line) < 50):
                        potential_headers.append(line)
                    
                    # Potential data (contains numbers, short)
                    elif (any(char.isdigit() for char in line) and 
                          len(words) <= 10 and len(line) < 80):
                        potential_data.append(line)
                
                # Try to reconstruct table structure
                if potential_headers and potential_data:
                    # Use headers as column names
                    if len(potential_headers) >= 2:
                        table_data.append(potential_headers[:5])  # Max 5 columns
                    
                    # Group data into rows
                    for data_line in potential_data[:10]:  # Max 10 data rows
                        words = data_line.split()
                        if len(words) >= 1:
                            table_data.append(words[:5])  # Max 5 columns
                
                if len(table_data) >= 2:  # At least header + 1 data row
                    tables.append({
                        'page': page_num + 1,
                        'table_index': len(tables),
                        'data': table_data,
                        'bbox': None,
                        'extraction_method': 'reconstructed',
                        'title': line
                    })
                    print(f"Reconstructed table with {len(table_data)} rows")
                    
                    # Show sample of reconstructed table
                    for row_idx, row in enumerate(table_data[:3]):
                        print(f"  Row {row_idx}: {row}")
                
                i = j  # Move past this table
            else:
                i += 1
    
    doc.close()
    return tables

# Use the simple version if layoutparser is not available
def extract_text_and_layout(pdf_path):
    try:
        import layoutparser as lp
        return extract_text_with_layoutparser(pdf_path)
    except Exception as e:
        print("layoutparser not available or failed:", str(e))
        print("Using simplified extraction...")
        return extract_text_and_layout_simple(pdf_path)