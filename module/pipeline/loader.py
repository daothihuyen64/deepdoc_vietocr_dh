from PIL import Image


def load_pdf_pages(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    from pdf2image import convert_from_path
    return convert_from_path(pdf_path, dpi=dpi)
