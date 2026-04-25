import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import urllib.request
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


def compute_image_stats(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 255.0
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(laplacian_var / 500.0, 1.0)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    return brightness, contrast, blur_score, edge_density


def compute_green_ratio(image_cv):
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = np.sum(mask > 0)
    total_pixels = image_cv.shape[0] * image_cv.shape[1]
    return green_pixels / total_pixels


def compute_color_entropy(image_cv):
    hist = cv2.calcHist([image_cv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy / 24.0


places_model = None
places_categories = None

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Скачивание {dest}...")
        urllib.request.urlretrieve(url, dest)
        print("Готово.")

def load_places_model():
    global places_model, places_categories
    if places_model is None:
        labels_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        labels_file = 'categories_places365.txt'
        download_file(labels_url, labels_file)
        with open(labels_file, 'r') as f:
            places_categories = [line.strip().split(' ')[0][3:] for line in f]

        weights_url = 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
        weights_file = 'resnet18_places365.pth.tar'
        download_file(weights_url, weights_file)

        model = models.resnet18(num_classes=365)
        checkpoint = torch.load(weights_file, map_location='cpu')
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        places_model = model
    return places_model

def get_scene_label(image_pil, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_idx = torch.topk(probs, 1)
    return places_categories[top_idx.item()], top_prob.item()


def load_detection_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

FURNITURE_CLASSES = {
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    72: 'refrigerator', 73: 'oven', 74: 'sink', 75: 'microwave',
    76: 'toaster', 77: 'hair drier', 78: 'toothbrush'
}

def count_furniture_items(predictions, score_thresh=0.5):
    """Возвращает количество объектов мебели (сумма по всем классам мебели)."""
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']
    count = 0
    for box, score, label in zip(boxes, scores, labels):
        if score > score_thresh and label.item() in FURNITURE_CLASSES:
            count += 1
    return count

def count_objects_general(predictions, score_thresh=0.5):
    """Общее количество любых объектов (для обратной совместимости)."""
    scores = predictions['scores']
    return int((scores > score_thresh).sum().item())


def estimate_wall_floor_ratio(image_cv):
    """Отношение площади стен к полу (упрощённо: верхние 70% / нижние 30%)."""
    h, w = image_cv.shape[:2]
    floor_region = image_cv[int(h*0.7):, :]
    wall_region = image_cv[:int(h*0.7), :]
    wall_pixels = wall_region.size
    floor_pixels = floor_region.size
    ratio = wall_pixels / (floor_pixels + 1)
    return np.clip(ratio, 0, 10)

def compute_light_uniformity(image_cv):
    """Равномерность освещения: 1 - (std / mean) нормализованно."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    std = np.std(gray)
    if mean < 1e-6:
        return 1.0
    uniformity = 1 - (std / (mean + 1e-6))
    return np.clip(uniformity, 0, 1)

def detect_windows_simple(image_cv):
    """Грубое детектирование окон по ярким прямоугольным областям."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    windows = 0
    h_img, w_img = gray.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        area = w * h
        if area > 0.02 * h_img * w_img and 0.4 < aspect < 2.5:
            windows += 1
    return windows

def compute_defect_score(image_cv):
    """
    Простой дефект-скоринг на основе градиентов в тёмных областях.
    Чем выше значение, тем больше подозрительных областей (трещины, сколы).
    """
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = grad_mag / grad_mag.max() 
    dark_mask = gray < 80
    defect_mask = (grad_mag > 0.3) & dark_mask
    defect_ratio = np.sum(defect_mask) / (gray.shape[0] * gray.shape[1])
    return np.clip(defect_ratio * 5, 0, 1)  # макс 1

def compute_furniture_density(num_furniture, area_pixels):
    """Плотность мебели = количество мебели / (площадь изображения в млн пикселей)."""
    return num_furniture / (area_pixels / 1_000_000 + 1e-6)


def analyze_photo(image_path, location_name, scene_model, detection_model):
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise ValueError(f"Не удалось загрузить: {image_path}")
    image_pil = Image.open(image_path).convert('RGB')

    filename = os.path.basename(image_path)
    scene_label, _ = get_scene_label(image_pil, scene_model)
    brightness, contrast, blur_score, edge_density = compute_image_stats(image_cv)
    green_ratio = compute_green_ratio(image_cv)
    color_entropy = compute_color_entropy(image_cv)

    transform_det = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform_det(image_pil).unsqueeze(0)
    with torch.no_grad():
        predictions = detection_model(img_tensor)[0]
    num_total_objects = count_objects_general(predictions)
    num_furniture = count_furniture_items(predictions)

    wall_floor_ratio = estimate_wall_floor_ratio(image_cv)
    light_uniformity = compute_light_uniformity(image_cv)
    num_windows = detect_windows_simple(image_cv)
    defect_score = compute_defect_score(image_cv)
    area_pixels = image_cv.shape[0] * image_cv.shape[1]
    furniture_density = compute_furniture_density(num_furniture, area_pixels)

    aesthetic = compute_aesthetic_score(brightness, blur_score, edge_density, green_ratio)

    return {
        'filename': filename,
        'location': location_name,
        'scene_category': scene_label,
        'brightness': round(brightness, 4),
        'contrast': round(contrast, 4),
        'blur_score': round(blur_score, 4),
        'edge_density': round(edge_density, 4),
        'num_objects': num_total_objects,
        'green_ratio': round(green_ratio, 4),
        'color_entropy': round(color_entropy, 4),
        'wall_floor_ratio': round(wall_floor_ratio, 4),
        'light_uniformity': round(light_uniformity, 4),
        'num_windows': num_windows,
        'defect_score': round(defect_score, 4),
        'furniture_count': num_furniture,
        'furniture_density': round(furniture_density, 4),
        'aesthetic_score': round(aesthetic, 4)
    }

def compute_aesthetic_score(brightness, blur_score, edge_density, green_ratio):
    score = (brightness * 0.4 +
             (1 - blur_score) * 0.3 +
             (1 - min(edge_density, 0.3)) * 0.2 +
             green_ratio * 0.1)
    return min(max(score, 0), 1)


def build_dataset_from_images(root_folder, output_csv='dataset.csv'):
    scene_model = load_places_model()
    detection_model = load_detection_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scene_model.to(device)
    detection_model.to(device)

    data_rows = []
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp')

    for location_name in os.listdir(root_folder):
        location_path = os.path.join(root_folder, location_name)
        if not os.path.isdir(location_path):
            continue
        for fname in os.listdir(location_path):
            if fname.lower().endswith(supported_ext):
                img_path = os.path.join(location_path, fname)
                try:
                    row = analyze_photo(img_path, location_name, scene_model, detection_model)
                    data_rows.append(row)
                    print(f"Обработано: {location_name}/{fname}")
                except Exception as e:
                    print(f"Ошибка {location_name}/{fname}: {e}")

    if not data_rows:
        print("Не найдено ни одного изображения в подпапках.")
        return None

    df = pd.DataFrame(data_rows)
    df.to_csv(output_csv, index=False)
    print(f"Датасет сохранён в {output_csv} (строк: {len(df)})")
    print(f"Колонки: {list(df.columns)}")
    return df


if __name__ == "__main__":
    ROOT_FOLDER = "./photos_example"
    if not os.path.exists(ROOT_FOLDER):
        os.makedirs(ROOT_FOLDER)
        print(f"Создана папка {ROOT_FOLDER}. Внутри неё создайте подпапки (backyard, bathroom и т.д.) и поместите туда фото.")
    else:
        dataset = build_dataset_from_images(ROOT_FOLDER, "room_dataset.csv")
        if dataset is not None:
            print("\nПервые 5 строк датасета:")
            print(dataset.head())