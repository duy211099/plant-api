from flask import Flask, request, jsonify

from torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        file=request.files.get('file')
        if file is None or file.filename== "":
            return jsonify({'result':'no file'})
        if not allowed_file(file.filename):
            return jsonify({'result':'format not spted'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction=get_prediction(tensor)
            print(prediction[0])
            return jsonify({
            'status': 'success',
            'accuracy': str(prediction[0]),
            'label': str(prediction[1])
        })     
        except:
            return jsonify({
            'status': 'fail'
        })     


if __name__ == '__main__':
    app.run(debug=True)

# set FLASK_APP=main.py
# $env:FLASK_APP = "main.py"
# set FLASK_ENV=development
# flask run