from PIL import Image

def to_images(file_path):
  import fitz
  doc = fitz.open(file_path)
  images = []
  for index in range(0, doc.page_count):
    page = doc.load_page(index)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    images.append(img)
  return images