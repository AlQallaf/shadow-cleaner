from flask import Flask, request, send_file
import fitz  # PyMuPDF
import cv2
import numpy as np
import os
from PIL import Image
import uuid
import io

app = Flask(__name__)

def remove_shadows(gray_img):
    """
    Flatten uneven lighting while keeping original glyph detail.
    Returns a high-contrast grayscale image with a white background.
    """
    denoised = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)
    background = cv2.medianBlur(denoised, 31)
    normalized = cv2.divide(denoised, background, scale=255)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    text_mask = cv2.bitwise_not(binary)
    text_detail = cv2.bitwise_and(gray_img, text_mask)
    cleaned = cv2.max(text_detail, binary)
    cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)
    return cleaned

def clean_image_background(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return remove_shadows(gray)

def convert_images_to_pdf(image_files, output_pdf_path):
    image_list = []
    for image_path in image_files:
        cleaned = clean_image_background(image_path)
        pil_img = Image.fromarray(cleaned).convert("RGB")
        image_list.append(pil_img)

    if image_list:
        image_list[0].save(output_pdf_path, save_all=True, append_images=image_list[1:])

def clean_pdf_background(input_path, output_path, zoom=2):
    doc = fitz.open(input_path)
    output_doc = fitz.open()
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cleaned = remove_shadows(gray)
            # Encode to PNG bytes because PyMuPDF expects a standard image stream.
            success, png_buffer = cv2.imencode(".png", cleaned)
            if not success:
                raise ValueError("Failed to encode cleaned page to PNG")
            new_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(
                new_page.rect,
                stream=png_buffer.tobytes(),
            )

        output_doc.save(output_path)
    finally:
        output_doc.close()
        doc.close()

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_files = request.files.getlist('file')
    if not uploaded_files:
        return "No files uploaded", 400

    unique_id = str(uuid.uuid4())
    input_pdf = f"temp_input_{unique_id}.pdf"
    output_pdf = f"temp_output_{unique_id}.pdf"
    temp_targets = []

    try:
        if uploaded_files[0].filename.lower().endswith(".pdf"):
            uploaded_files[0].save(input_pdf)
            temp_targets.append(input_pdf)
        else:
            image_paths = []
            for i, file in enumerate(uploaded_files):
                img_path = f"temp_img_{unique_id}_{i}.jpg"
                file.save(img_path)
                image_paths.append(img_path)

            convert_images_to_pdf(image_paths, input_pdf)
            temp_targets.append(input_pdf)

            # Clean up original images
            for img in image_paths:
                os.remove(img)

        clean_pdf_background(input_pdf, output_pdf)
        download_name = f"cleaned_{unique_id}.pdf"

        with open(output_pdf, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        response = send_file(
            io.BytesIO(pdf_bytes),
            as_attachment=True,
            download_name=download_name,
            mimetype="application/pdf",
        )

        temp_targets.append(output_pdf)
        for path in temp_targets:
            if os.path.exists(path):
                continue;
           #     os.remove(path)

        return response

    except Exception as e:
        print("Error:", e)
        for path in temp_targets:
            if os.path.exists(path):
                os.remove(path)
        return "Error processing file(s)", 500

@app.route('/')
def hello():
    return "🧼 PDF/Image Background Cleaner API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
