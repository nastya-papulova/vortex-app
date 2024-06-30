import streamlit as st
from PIL import Image
import base64
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F

IMAGE_SIZE = (128, 128)
CLASS_NAMES = [0, 2, 4, 6, 8]


def classify(image, model, class_names, transform, image_size, device):
    # преобразование в тензор
    tensor = transform(image)
    tensor = tensor.to(device)
    tensor = tensor.unsqueeze_(0)

    # получение предсказаний
    model.eval()
    output = model(tensor)
    probs = F.softmax(output.to('cpu'), dim=1).detach().numpy()

    class_name = class_names[np.argmax(probs)]
    confidence_score = probs.max()

    return class_name, confidence_score


def set_bacground(image_file):
    # вставляет изображение как фон
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    return style


def set_interactive_elements(style):
    st.markdown(style, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>Определение топологического заряда оптических вихрей</h1>",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; font-size: 24px;'>Загрузите изображение интерференционной картины</h2>",
                unsafe_allow_html=True)

    # загрузчик файлов
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'],
                            help='Загрузите изображение интерференционной картины плоской волны и оптического вихря с зарядом 0, 2, 4, 6, 8. Предпочтительно, чтобы изображение было квадратным.')
    return file


background_file = 'bg.jpg'
style = set_bacground(background_file)
file = set_interactive_elements(style)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load("../../models/best_model.pth")

transform_image = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.4], std=[0.7])
])

# отображает и классифицирует изображение
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    max_class, max_prob = classify(image, model, CLASS_NAMES, transform_image, IMAGE_SIZE, device)

    st.write("## Топологичекий заряд равен {}".format(max_class))
    st.write("### Вероятность: {}%".format(int(max_prob * 100)))
