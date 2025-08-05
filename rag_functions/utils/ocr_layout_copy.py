import layoutparser as lp
from pdf2image import convert_from_path
from layoutparser.elements import Layout
import re 

import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Functions ############################################################################################################
## Identify Text Areas and extract
def inflate_layout(layout, top=0, bottom=0, left=0, right=0):
    new_blocks = []
    for block in layout:
        x1, y1, x2, y2 = block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2

        # Apply padding
        new_x1 = max(0, x1 - left)
        new_y1 = max(0, y1 - top)
        new_x2 = x2 + right
        new_y2 = y2 + bottom

        new_block = lp.TextBlock(
            lp.Rectangle(new_x1, new_y1, new_x2, new_y2),
            type=block.type,
            text=block.text,
            id=block.id,
            score=block.score,
        )
        new_blocks.append(new_block)

    return lp.Layout(new_blocks)

def compute_iou(rect1, rect2):
    # Coordinates of intersection box
    x_left = max(rect1.x_1, rect2.x_1)
    y_top = max(rect1.y_1, rect2.y_1)
    x_right = min(rect1.x_2, rect2.x_2)
    y_bottom = min(rect1.y_2, rect2.y_2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = rect1.area + rect2.area - intersection_area

    return intersection_area / union_area

def remove_mostly_overlapping_boxes(layout, iou_threshold=0.8):
    filtered = []

    for i, box1 in enumerate(layout):
        keep = True
        for j, box2 in enumerate(layout):
            if i == j:
                continue

            if box1.type == box2.type and box1.block.area < box2.block.area:
                iou = compute_iou(box1.block, box2.block)
                if iou > iou_threshold:
                    keep = False
                    break

        if keep:
            filtered.append(box1)

    return lp.Layout(filtered)

# Sort text chunks
def is_two_column(layout, page_width, threshold=0.15, min_ratio=0.2):
    """
    Detects whether a layout follows a two-column format.
    """
    total_blocks = len(layout)
    if total_blocks == 0:
        return False  # Can't be two-column if there are no blocks

    center = page_width / 2
    left_band_max = center - (threshold * page_width)
    right_band_min = center + (threshold * page_width)

    left_blocks = []
    right_blocks = []

    for b in layout:
        x_center = (b.coordinates[0] + b.coordinates[2]) / 2
        if x_center < left_band_max:
            left_blocks.append(b)
        elif x_center > right_band_min:
            right_blocks.append(b)

    left_ratio = len(left_blocks) / total_blocks
    right_ratio = len(right_blocks) / total_blocks

    return left_ratio > min_ratio and right_ratio > min_ratio



def sort_blocks_by_layout(blocks, page_width):
    if is_two_column(blocks, page_width):
        # Sort first by left/right column, then top to bottom
        left = [b for b in blocks if b.coordinates[0] < page_width / 2]
        right = [b for b in blocks if b.coordinates[0] >= page_width / 2]

        left_sorted = Layout(left).sort(key=lambda b: b.coordinates[1])
        right_sorted = Layout(right).sort(key=lambda b: b.coordinates[1])

        return left_sorted + right_sorted
    else:
        # Single-column: sort top to bottom, then left to right
        return Layout(blocks).sort(key=lambda b: (b.coordinates[1], b.coordinates[0]))
    

## Final text extraction helper functions 
def id_table(text):
    pattern = r'^Table \d+:'
    return bool(re.match(pattern, text))


# Global model variable - will be loaded when needed
model = None

def _load_layout_model():
    """Load the layout detection model lazily"""
    global model
    if model is None:
        model_path = '/Users/alexander/Library/CloudStorage/Dropbox/Personal Research/cortex_bridge/rag_functions/models/model_final.pth'
        config_path = 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config'
        model = lp.Detectron2LayoutModel(
            config_path,
            model_path=model_path,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        )
    return model


# Primary Function ############################################################################################################
def extract_text_and_layout(pdf_path):
    """Extract text and layout from PDF using layout detection"""
    layout_model = _load_layout_model()
    
    # Convert PDF to images
    pages = convert_from_path(pdf_path, dpi=300, poppler_path="/opt/homebrew/bin")

    total_text = ""
    for i, image in enumerate(pages):
        # Detect layout elements
        layout = layout_model.detect(image)
        
        # Process layout: inflate boxes and remove overlaps
        layout = inflate_layout(layout, top=15, bottom=3, left=6, right=6)
        layout = remove_mostly_overlapping_boxes(layout, iou_threshold=0.5)
        
        # Sort text blocks by reading order
        page_width, page_height = image.size
        blocks = [b for b in layout if b.type in ("Title", "Text")]
        layout_sorted = sort_blocks_by_layout(blocks, page_width)

        # Extract text into sections
        sections = []
        current_section = {"heading": "Document", "content": []}

        for block in layout_sorted:
            cropped = image.crop(block.coordinates)
            text = pytesseract.image_to_string(cropped).strip()
            if not text:
                continue

            if block.type == "Title":
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"heading": text, "content": []}
            else:
                current_section["content"].append(text)

        # Add final section
        if current_section["content"]:
            sections.append(current_section)

        # Format page text
        page_text = ""
        for section in sections:
            if section['heading'] != 'Document': 
                page_text += f"# {section['heading']} \n"
            for paragraph in section["content"]:
                if not id_table(paragraph) and not paragraph.startswith("Note. "):
                    page_text += paragraph + "\n \n"
        
        total_text += page_text

    return total_text

