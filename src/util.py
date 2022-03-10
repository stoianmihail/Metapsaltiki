from PIL import Image
import numpy as np

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

def last_arg(xs, fn):
  return len(xs) - fn(xs[::-1]) - 1