import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models


class BoneAgeModel(nn.Module):
    def __init__(self):
        super(BoneAgeModel, self).__init__()
        
        # Load ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify ResNet’s final FC layer to output 128 features
        self.resnet.fc = nn.Linear(512, 128)

        # New fully connected layer: (128 image features + 1 gender) → 1 bone age output
        self.fc = nn.Linear(128 + 1, 1)

    def forward(self, images, gender):
        x = self.resnet(images)  # Get image features
        gender = gender.view(-1, 1)  # Reshape gender to (batch_size, 1)
        x = torch.cat((x, gender), dim=1)  # Concatenate gender with image features
        x = self.fc(x)  # Final bone age prediction
        return x
#####################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model architecture
model = BoneAgeModel()

# Ensure model is loaded correctly, whether on CPU or GPU
model.load_state_dict(torch.load("bone_age_checkpoint_epoch_30.pth", map_location=device))

# Move model to the appropriate device
model.to(device)

# Set to evaluation mode
model.eval()
######################################################################################################

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # forr resnet18
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust mean/std if needed
])


#######################################################################################################

def predict_bone_age(image_path, gender):
    """ Predict bone age from an X-ray image and gender (0=Female, 1=Male). """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Convert gender (0=Female, 1=Male) to tensor
    gender_tensor = torch.tensor([gender], dtype=torch.float32).to(device)

    # Perform inference
    with torch.no_grad():
        prediction = model(image, gender_tensor)
        predicted_bone_age = prediction.item() / 12  # Convert months to years

    return round(predicted_bone_age, 2)