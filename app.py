from flask import Flask, render_template, request
import os

from detector.poster_detector import detect_poster

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

@app.route("/")
def home():
    return render_template("demo.html")

@app.route("/process", methods=["POST"])
def process():
    image = request.files["image"]

    filepath = os.path.join(
        UPLOAD_FOLDER,
        image.filename
    )

    image.save(filepath)

    print("Image received:", image.filename)

    # logo_found = detect_logo(filepath)
    corners = detect_poster(filepath)
    poster_found = corners is not None
    
    return render_template(
        "result.html",
        image_name=image.filename,
        poster_found=poster_found
    )

if __name__ == "__main__":
    app.run(debug=True)