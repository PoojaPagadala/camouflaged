from flask import Flask, render_template, request
from model.model import load_model, predict_image
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

# Load the model once
processor, model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_path = None
    predicted_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_path)

            image = Image.open(uploaded_path).convert("RGB")
            mask = predict_image(image, processor, model)

            predicted_path = os.path.join(app.config['OUTPUT_FOLDER'], f"mask_{filename}")
            mask.save(predicted_path)

            # Add leading slashes so URLs are correctly rendered in HTML
            uploaded_path = "/" + uploaded_path
            predicted_path = "/" + predicted_path

    return render_template("index.html", uploaded_path=uploaded_path, predicted_path=predicted_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
