import streamlit as st
from torch import nn
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from TinyVGG import tinyVGG
from matplotlib import pyplot as plt
from PIL import Image


class_names= ['AFRICAN LEOPARD','CARACAL','CHEETAH','CLOUDED LEOPARD','JAGUAR','LIONS','OCELOT','PUMA','SNOW LEOPARD','TIGER']

eff_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

inc_transform =transforms.Compose([
    transforms.Resize(size=(342,342)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

vgg_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor() 
])

def effnet():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Sequential(
    nn.BatchNorm1d(num_features=1280),    
    nn.Linear(1280, 512),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=512),
    nn.Dropout(0.4),
    nn.Linear(512, 10),
    )
    model.load_state_dict(torch.load('Best_model/EfficientNet B0/EfficientNet_B0_catClassifier.pth',map_location=torch.device('cpu')))
    return model


def inception():
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights = weights)
    model.fc = nn.Linear(in_features = 2048, out_features = 10)
    model.load_state_dict(torch.load('Best_model/Inception/Inception_catClassifier.pth',map_location=torch.device('cpu')))
    return model

def tinyvgg1():
    model = tinyVGG(input_shape=3,hidden_units=10,output_shape=10)
    model.load_state_dict(torch.load('Best_model/tinyvgg/TinyVGGmodel.pth',map_location=torch.device('cpu')))
    return model

def predict(model,img):
    model.eval()
    with torch.inference_mode():
        img = img.unsqueeze(dim=0)
        target_image_pred = model(img)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    return target_image_pred_label
    
st.image('https://c1.wallpaperflare.com/preview/987/480/945/big-cats-collage-predators-animals.jpg',width=350)
st.header('Wild Big Cat Classifier')
st.write("Welcome to my project where you can explore the fascinating world of wild cats through the lens of cutting-edge deep learning technology. Trained  on different types of neural networks, including EfficientNet, Inception, and TinyVGG, on a comprehensive dataset of wild cats, enabling you to easily identify various feline species. Whether you're a wildlife enthusiast or simply a cat lover, you're sure to enjoy this exciting journey into the wild. So, join us and let's discover the beauty and diversity of wild cats together!")


with st.sidebar:
    st.image('https://miro.medium.com/v2/resize:fit:1400/1*3fA77_mLNiJTSgZFhYnU0Q.png')
    st.title('Choose Neural Network')
    selection = st.selectbox('Choose the Neural Network: ',options=('EfficientNet B0','Inception','TinyVGG'),help='Choose model for prediction')
    st.write('Test Accuracy of different Models:')
    st.markdown("EfficientNet B0: 96%")
    st.markdown("Inception v3: 95%")
    st.markdown("TinyVGG: 56%")

if selection == 'EfficientNet B0':
    st.sidebar.image('https://www.researchgate.net/publication/349299852/figure/fig3/AS:991096909869056@1613307325251/The-network-architecture-of-EfficientNet-It-can-output-a-feature-map-with-deep-semantic.jpg',caption='EfficientNet structure')
    st.markdown('#### EfficientNet B0')
    st.write('EfficientNet is a family of convolutional neural networks designed to achieve state-of-the-art accuracy with fewer parameters and faster inference times. EfficientNet models have been widely adopted in various computer vision tasks, such as image classification, object detection, and segmentation.')

    st.markdown('#### Upload Image')
    file = st.file_uploader('choose Image:',type=['jpg','jpeg','png'])
    if file:
        img = Image.open(file)
        transimg = eff_transform(img)
        transimg_modified = transimg.permute(1, 2, 0).numpy()
        st.markdown('##### Uploaded image')
        st.image(file)
        st.markdown('##### Transformed Image')
        st.image(transimg_modified,clamp=True,channels='RGB')

        st.markdown('#### Prediction')
        st.subheader(body=class_names[predict(model=effnet(),img=transimg)])


if selection == 'Inception':
    st.sidebar.image('https://www.researchgate.net/publication/349717475/figure/fig5/AS:996933934014464@1614698980419/The-architecture-of-Inception-V3-model.ppm',caption='Inception structure')
    st.markdown('#### Inception V3')
    st.write('Inception-v3 is a convolutional neural network architecture that was introduced by Google in 2015. It is a deep neural network with 48 layers that achieved state-of-the-art performance on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) classification task. Inception-v3 is known for its use of factorization in convolutional layers, which reduces the number of computations required during training and inference.')

    st.markdown('#### Upload Image')
    file = st.file_uploader('choose Imge:',type=['jpg','jpeg','png'])
    if file:
        img = Image.open(file)
        transimg = inc_transform(img)
        transimg_modified = transimg.permute(1, 2, 0).numpy()
        st.markdown('##### Uploaded image')
        st.image(file)
        st.markdown('##### Transformed Image')
        st.image(transimg_modified,clamp=True,channels='RGB')

        st.markdown('#### Prediction')
        st.subheader(body=class_names[predict(model=inception(),img=transimg)])


if selection == 'TinyVGG':
    st.sidebar.image('https://nnart.org/wp-content/uploads/2022/06/vgg-a-banner.webp',caption='TinyVGG structure')
    st.markdown('#### TinyVGG')
    st.write('TinyVGG is a compact version of the popular VGG network architecture, designed to achieve high accuracy with fewer parameters. It was introduced by researchers at the University of Oxford in 2017. TinyVGG consists of only a few layers compared to the original VGG, making it more efficient to train and faster to run on low-power devices.')

    st.markdown('#### Upload Image')
    file = st.file_uploader('choose Imge:',type=['jpg','jpeg','png'])
    if file:
        img = Image.open(file)
        transimg = vgg_transform(img)
        transimg_modified = transimg.permute(1, 2, 0).numpy()
        st.markdown('##### Uploaded image')
        st.image(file)
        st.markdown('##### Transformed Image')
        st.image(transimg_modified,clamp=True,channels='RGB')

        st.markdown('#### Prediction')
        st.subheader(body=class_names[predict(model=tinyvgg1(),img=transimg)])