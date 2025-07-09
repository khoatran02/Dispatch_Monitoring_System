import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

class Classifier:
    def __init__(self, model_name='MobileNetV3', data_path='Classification/dish', num_classes=3):
        """
        Initialize the dish classifier with proper image resizing for each model.
        
        Args:
            model_name (str): Name of the model (MobileNetV3, ResNet50, etc.)
            data_path (str): Path to the dataset folder with structure:
                            ├── dish/
                            │   ├── empty/
                            │   ├── kakigori/
                            │   └── not_empty/
            num_classes (int): Number of output classes
        """
        self.model_name = model_name
        self.data_path = data_path
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ['empty', 'kakigori', 'not_empty']
        
        # Model-specific input sizes
        self.model_input_sizes = {
            'mobilenetv3': 224,
            'resnet18': 224,
            'resnet50': 224,
            'efficientnet_b3': 300  # EfficientNet typically uses 300x300
        }
        
        # Initialize transforms with model-specific sizing
        input_size = self.model_input_sizes.get(model_name.lower(), 224)
        self.transform = self._get_transforms(input_size)
        
        # Initialize model
        self.model = self._initialize_model()
        self.model = self.model.to(self.device)
        
    def _get_transforms(self, input_size):
        """Create transforms with proper resizing and augmentation"""
        return transforms.Compose([
            transforms.Resize(input_size),  # First resize maintaining aspect ratio
            transforms.CenterCrop(input_size),  # Then center crop
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _initialize_model(self):
        """Initialize the specified model with pretrained weights"""
        model_name = self.model_name.lower()
        
        if model_name == 'mobilenetv3':
            model = models.mobilenet_v3_small(pretrained=True)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
        
        return model
    
    def load_data(self, batch_size=32, val_split=0.2):
        """Load and split the dataset into train and validation sets"""
        full_dataset = datasets.ImageFolder(self.data_path, transform=self.transform)
        
        # Calculate split sizes
        val_size = int(val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        # Split dataset
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Dataset loaded: {train_size} training samples, {val_size} validation samples")

    def save_model(self, path=None):
        """Save the current model state"""
        if path is None:
            path = f"{self.model_name.lower()}_dish_classifier.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'class_names': self.class_names
        }, path)
        print(f"Model saved to {path}")
        
    def train(self, epochs=10, lr=0.001):
        """Train the model with progress tracking"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(self.train_loader, desc="Training"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
            
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Validation phase
            val_loss, val_acc = self.evaluate()
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f'best_{self.model_name}.pth')
        
        print(f'Training complete. Best val Acc: {best_acc:.4f}')
    
    def evaluate(self):
        """Evaluate the model on validation set"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        loss = running_loss / len(self.val_loader.dataset)
        acc = running_corrects.double() / len(self.val_loader.dataset)
        
        return loss, acc
    
    def load_model(self, model_path):
        """Load a saved model from path"""
        checkpoint = torch.load(model_path)
        
        # Verify model compatibility
        if 'model_name' in checkpoint and checkpoint['model_name'] != self.model_name:
            print(f"Warning: Loading {checkpoint['model_name']} into {self.model_name}")
        
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval().to(self.device)  # Set to evaluation mode
        print(f"Model loaded from {model_path}")

        return self

    def predict(self, image, return_prob=False):
        """
        Predict class for a single image
        
        Args:
            image: Can be:
                - PIL Image
                - OpenCV BGR image (numpy array)
                - File path string
            return_prob: If True, returns class probabilities
        
        Returns:
            Predicted class name (or probabilities if return_prob=True)
        """
        # Convert input to PIL Image if needed
        if isinstance(image, str):  # File path
            image = Image.open(image)
        elif isinstance(image, np.ndarray):  # OpenCV image
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        if return_prob:
            return {cls: float(prob) for cls, prob in zip(self.class_names, probs[0])}
        else:
            return self.class_names[probs.argmax().item()]

