from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn


# Определение преобразований, которые были использованы для обучения модели
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Загрузка изображения
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Конвертация изображения в RGB
    image = img_transforms(image)
    image = image.unsqueeze(0)  # Добавление дополнительного измерения для батча
    return image

# Функция для предсказания
def predict(model, image_path, device):
    model.eval()  # Перевод модели в режим оценки
    image = load_image(image_path)
    image = image.to(device)
    with torch.no_grad():  # Отключение автоматического вычисления градиентов для экономии памяти
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    return predicted_class.item(), confidence.item()

# Загрузка модели
model = SimpleCNN()
model.load_state_dict(torch.load("data_science/save_model/saved_model.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Пример использования модели для предсказания на новом изображении
image_path = "data_science/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg"
predicted_class, confidence = predict(model, image_path, device)
print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
