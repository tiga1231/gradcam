import base64
from PIL import Image
from io import BytesIO


from gradcam import compute_gradcam



import torch
from torchvision import transforms

from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin


# print(
#     compute_gradcam(torch.randn(3,224,224))
# )





test_fn = 'cat-dog.jpg'
test_image = Image.open(test_fn)
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
])
to_img = transforms.ToPILImage()
test_image = img_transform(test_image)


buffered = BytesIO()
to_img(test_image).save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode('ascii')



app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/upload_image', methods=['GET', 'POST'])
@cross_origin()
def upload_image():
    global test_image, img_str
    if request.method == 'POST':
        req = request.get_json(force=True)
        img_data = req['data'][len('data:image/jpeg;base64,'):]
        
        test_image = Image.open(BytesIO(base64.b64decode(img_data)))
        test_image = img_transform(test_image)

        buffered = BytesIO()
        to_img(test_image).save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('ascii')

        return 'got it'
    return '<p>Hello, World!</p>'


@app.route('/gradcam', methods=['GET', 'POST'])
@cross_origin()
def post_gradcam():
    if request.method == 'POST':
        req = request.get_json(force=True)
        gradcam = compute_gradcam(test_image, int(req['class_index']))
        # print(gradcam)
        req['gradcam'] = gradcam.numpy().tolist()
        req['image'] = img_str
        return req
    return '<p>Hello, World!</p>'