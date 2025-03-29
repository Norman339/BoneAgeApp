from flask import Flask, render_template, request
from appmodel import predict_bone_age
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["file"]
        gender = int(request.form["gender"])  # Get gender input (0=Female, 1=Male)

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)  # Save uploaded file
            
            # Get bone age prediction
            prediction = predict_bone_age(file_path, gender)

            # Remove the file after prediction (optional)
            os.remove(file_path)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
