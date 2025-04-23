from flask import Flask, request, render_template_string
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        result = classify_stool(filepath)
        return f"<h1>Result: {result}</h1><br><a href='/'>Try another</a>"
    
    return '''
    <h1>Upload Stool Image for Classification</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit">
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)