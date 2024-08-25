# In[1]:
import os
import itertools
from PIL import Image
from torchvision import transforms

# Base directory where original images are stored
base_dir = 'C:/Users/ARA/Desktop/ara330/Outfit_Aura/static/images/'

# Directory to save processed images
processed_dir = 'C:/Users/ARA/Desktop/ara330/Outfit_Aura/static/images/processed'

# Create the processed image directory structure
def create_dir_structure(base_dir, processed_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            # Create corresponding directory in the processed folder
            dir_path = os.path.join(processed_dir, os.path.relpath(os.path.join(root, dir_name), base_dir))
            os.makedirs(dir_path, exist_ok=True)

create_dir_structure(base_dir, processed_dir)

# Image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to denormalize the image before saving (for visualization)
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Reverse the normalization
    return tensor

# Function to load, preprocess, and save images
def load_preprocess_and_save_image(image_path, save_path):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)
    
    # Denormalize before saving for correct visualization
    denorm_tensor = denormalize(tensor.clone(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Convert back to PIL image for saving
    save_image = transforms.ToPILImage()(denorm_tensor)
    save_image.save(save_path)
    
    return tensor

# Function to load image paths from a given directory
def load_image_paths(root_dir):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Case-insensitive check
                image_paths.append(os.path.join(root, file))
    return image_paths

# Function to process images in a given category and save them in the corresponding processed directory
def process_images_in_category(category_dir, save_category_dir):
    image_paths = load_image_paths(category_dir)
    processed_images = []
    for image_path in image_paths:
        # Determine where to save the processed image
        relative_path = os.path.relpath(image_path, category_dir)
        save_path = os.path.join(save_category_dir, relative_path)
        
        # Process and save the image
        processed_images.append(load_preprocess_and_save_image(image_path, save_path))
    
    return processed_images

# Function to create pairs within two categories
def create_pairs(category1, category2):
    pairs = list(itertools.product(category1, category2))
    return pairs

# Process images in each category and save them
# Female
female_formal_bottoms = process_images_in_category(os.path.join(base_dir, 'female/formal/bottoms'), os.path.join(processed_dir, 'female/formal/bottoms'))
female_formal_tops = process_images_in_category(os.path.join(base_dir, 'female/formal/tops'), os.path.join(processed_dir, 'female/formal/tops'))

female_informal_bottoms = process_images_in_category(os.path.join(base_dir, 'female/informal/bottoms'), os.path.join(processed_dir, 'female/informal/bottoms'))
female_informal_tops = process_images_in_category(os.path.join(base_dir, 'female/informal/tops'), os.path.join(processed_dir, 'female/informal/tops'))

# Male
male_formal_bottoms = process_images_in_category(os.path.join(base_dir, 'male/formal/bottoms'), os.path.join(processed_dir, 'male/formal/bottoms'))
male_formal_tops = process_images_in_category(os.path.join(base_dir, 'male/formal/tops'), os.path.join(processed_dir, 'male/formal/tops'))

male_informal_bottoms = process_images_in_category(os.path.join(base_dir, 'male/informal/bottoms'), os.path.join(processed_dir, 'male/informal/bottoms'))
male_informal_tops = process_images_in_category(os.path.join(base_dir, 'male/informal/tops'), os.path.join(processed_dir, 'male/informal/tops'))



# In[10]:
import os
from PIL import Image
from torchvision import transforms
import itertools

# Directory where processed images are stored
processed_dir = 'C:/Users/ARA/Desktop/ara330/Outfit_Aura/static/images/'

# Image preprocessing transformations (same as used for processing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load images from a given directory
def load_images_from_dir(dir_path):
    image_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure only image files are loaded
                image_paths.append(os.path.join(root, file))
    
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image)
        images.append(tensor)
    
    return images

# Load the images from the processed directories
female_formal_tops = load_images_from_dir(os.path.join(processed_dir, 'female/formal/tops'))
female_formal_bottoms = load_images_from_dir(os.path.join(processed_dir, 'female/formal/bottoms'))

female_informal_tops = load_images_from_dir(os.path.join(processed_dir, 'female/informal/tops'))
female_informal_bottoms = load_images_from_dir(os.path.join(processed_dir, 'female/informal/bottoms'))

male_formal_tops = load_images_from_dir(os.path.join(processed_dir, 'male/formal/tops'))
male_formal_bottoms = load_images_from_dir(os.path.join(processed_dir, 'male/formal/bottoms'))

male_informal_tops = load_images_from_dir(os.path.join(processed_dir, 'male/informal/tops'))
male_informal_bottoms = load_images_from_dir(os.path.join(processed_dir, 'male/informal/bottoms'))

# Function to create pairs within two categories
def create_pairs(category1, category2):
    pairs = list(itertools.product(category1, category2))
    return pairs

# Recreate the pairs
female_formal_pairs = create_pairs(female_formal_tops, female_formal_bottoms)
female_informal_pairs = create_pairs(female_informal_tops, female_informal_bottoms)

male_formal_pairs = create_pairs(male_formal_tops, male_formal_bottoms)
male_informal_pairs = create_pairs(male_informal_tops, male_informal_bottoms)
"""
# Print the number of pairs created
print(f"Number of pairs created for female formal: {len(female_formal_pairs)}")
print(f"Number of pairs created for female informal: {len(female_informal_pairs)}")
print(f"Number of pairs created for male formal: {len(male_formal_pairs)}")
print(f"Number of pairs created for male informal: {len(male_informal_pairs)}")
"""
"""
# In[11]:
import random

# If there are too many pairs, you can randomly select a subset
# For example, let's say we want only 50,000 pairs out of the 16M+ pairs for female informal
desired_num_pairs = 50000
if len(female_informal_pairs) > desired_num_pairs:
    female_informal_pairs = random.sample(female_informal_pairs, desired_num_pairs)

print(f"Number of pairs after downsampling for female informal: {len(female_informal_pairs)}")

"""
# In[12]:
import random

# Example heuristic for synthetic labels
def generate_synthetic_labels(pairs):
    labels = []
    for (img1, img2) in pairs:
        # Simple heuristic: label pairs from the same category as 1 (compatible)
        # and from different categories as 0 (incompatible)
        if img1.shape == img2.shape:  # You might replace this condition with a more relevant feature
            labels.append(1)
        else:
            labels.append(0)
    return labels

# Generate synthetic labels for each of your pairs
female_formal_labels = generate_synthetic_labels(female_formal_pairs)
female_informal_labels = generate_synthetic_labels(female_informal_pairs)
male_formal_labels = generate_synthetic_labels(male_formal_pairs)
male_informal_labels = generate_synthetic_labels(male_informal_pairs)

"""
print(f"Number of female formal labels: {len(female_formal_labels)}")
print(f"Number of female informal labels: {len(female_informal_labels)}")
print(f"Number of male formal labels: {len(male_formal_labels)}")
print(f"Number of male informal labels: {len(male_informal_labels)}")
"""

# In[15]:
import torch

def create_tensors_from_pairs_in_batches(pairs, labels, batch_size):
    num_batches = len(pairs) // batch_size + (1 if len(pairs) % batch_size != 0 else 0)
    print(f"Number of batches: {num_batches}")  # Debug: Check the number of batches
    for i in range(num_batches):
        batch_pairs = pairs[i * batch_size:(i + 1) * batch_size]
        batch_labels = labels[i * batch_size:(i + 1) * batch_size]

        print(f"Processing batch {i+1}/{num_batches}")  # Debug: Indicate which batch is being processed
        
        pairs_tensor = torch.stack([pair_to_tensor(pair) for pair in batch_pairs])
        labels_tensor = torch.tensor(batch_labels, dtype=torch.float32)
        
        yield pairs_tensor, labels_tensor
"""
# Check the number of pairs and labels
print(f"Number of female formal pairs: {len(female_formal_pairs)}")
print(f"Number of female formal labels: {len(female_formal_labels)}")
"""
# Set your batch size here
batch_size = 100  # Adjust this depending on your system's memory


# In[16]:
import torch
import torch.nn as nn

# Define the Compatibility Model
class CompatibilityModel(nn.Module):
    def __init__(self):
        super(CompatibilityModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * 3 * 224 * 224, 512),  # Adjust input size based on image dimensions
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output between 0 and 1 for compatibility score
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.fc(x)

# Initialize the model
model = CompatibilityModel()

"""
# In[17]:
# Training loop
num_epochs = 10
batch_size = 100  # Adjust based on your memory capacity

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}/{num_epochs}")  # Debug: Indicate the start of an epoch
    for batch_pairs, batch_labels in create_tensors_from_pairs_in_batches(female_formal_pairs, female_formal_labels, batch_size):
        
        optimizer.zero_grad()
        outputs = model(batch_pairs)
        loss = criterion(outputs.squeeze(), batch_labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# In[18]:
import os
import torch

# Define the directory where you want to save the model
save_directory = '/C:/Users/ARA/Desktop/ara330/Outfit_Aura/static/models/'

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Define the full path for saving the model
model_filename = 'compatibility_model.pth'
model_path = os.path.join(save_directory, model_filename)


# In[19]:
# Assuming 'model' is your trained model instance
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")
"""

# In[20]:
import torch

# Initialize the model architecture
model = CompatibilityModel()

# Load the saved model weights
model_path = 'C:/Users/ARA/Desktop/ara330/Outfit_Aura/static/models/compatibility_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()


# In[22]:
import os
import torch
from PIL import Image
from torchvision import transforms

# Define the directory where preprocessed images are stored
preprocessed_dir = 'C:/Users/ARA/Desktop/ara330/Outfit_Aura/static/images/'

# Define the model architecture and load the trained model
model = CompatibilityModel()
model_path = 'C:/Users/ARA/Desktop/ara330/Outfit_Aura/static/models/compatibility_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Function to load a preprocessed image
def load_preprocessed_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()  # No need to resize or normalize since it's already preprocessed
    tensor = transform(image)
    return tensor

# Function to create a pair tensor
def create_pair_tensor(image1_tensor, image2_tensor):
    return torch.cat((image1_tensor.unsqueeze(0), image2_tensor.unsqueeze(0)), dim=1)

# Function to get the top 3 bottom recommendations for a selected top
def get_top_bottom_recommendations(selected_top_path, bottoms_dir, model, num_recommendations=3):
    # Load the selected top
    selected_top_tensor = load_preprocessed_image(selected_top_path)

    # Load all bottoms in the specified directory
    bottom_images = [os.path.join(bottoms_dir, file) for file in os.listdir(bottoms_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    scores = []
    for bottom_path in bottom_images:
        bottom_tensor = load_preprocessed_image(bottom_path)
        pair_tensor = create_pair_tensor(selected_top_tensor, bottom_tensor)
        with torch.no_grad():
            score = model(pair_tensor.unsqueeze(0)).item()
        scores.append((score, bottom_path))

    # Sort scores in descending order and select top 3
    scores.sort(reverse=True, key=lambda x: x[0])
    top_recommendations = scores[:num_recommendations]

    return top_recommendations
"""
# Example of usage
selected_top_path = os.path.join(preprocessed_dir, 'female/informal/tops/00029_00.png')
bottoms_dir = os.path.join(preprocessed_dir, 'female/informal/bottoms')

top_3_recommendations = get_top_bottom_recommendations(selected_top_path, bottoms_dir, model)

# Print the top 3 recommendations
for i, (score, bottom_path) in enumerate(top_3_recommendations, start=1):
    print(f"Recommendation {i}: {bottom_path} with score {score:.4f}")
"""

# In[23]:
import torch
import time
from PIL import Image
from torchvision import transforms

# Load the selected top image and preprocess it
def load_preprocessed_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    tensor = transform(image)
    return tensor

#selected_top_path = 'C:/Users/ARA/Desktop/AuraFit/static/images/female/informal/tops/00029_00.png'
#selected_top_tensor = load_preprocessed_image(selected_top_path)

bottoms_dir = 'C:/Users/ARA/Desktop/ara330/Outfit_Aura/static/images/male/formal/bottoms'
"""
# In[24]:
import os

# Directory containing the bottoms to compare against

model = CompatibilityModel()
model_path = 'C:/Users/ARA/Desktop/ara330/Outfit_Aura/static/models/compatibility_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
"""
# Compare the selected top with all available bottoms
def get_live_recommendations(selected_top_tensor, bottoms_dir, model, num_recommendations=3):
    bottom_images = [os.path.join(bottoms_dir, file) for file in os.listdir(bottoms_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    scores = []
    start_time = time.time()  # Start timing
    for bottom_path in bottom_images:
        bottom_tensor = load_preprocessed_image(bottom_path)
        pair_tensor = torch.cat((selected_top_tensor.unsqueeze(0), bottom_tensor.unsqueeze(0)), dim=1)
        with torch.no_grad():
            score = model(pair_tensor.unsqueeze(0)).item()
        scores.append((score, bottom_path))
    
    total_time = time.time() - start_time  # End timing
    
    # Sort scores in descending order and select top 3
    scores.sort(reverse=True, key=lambda x: x[0])
    top_recommendations = scores[:num_recommendations]
    
    return top_recommendations, total_time

def getSuggestions(input_image_path, type, gender,category):
    selected_top_tensor = load_preprocessed_image("C:/Users/ARA/Desktop/ara330/Outfit_Aura" + input_image_path)

    if category == "top":
        category = "bottoms"
    else:
        category = "tops"
    other_directory = "C:/Users/ARA/Desktop/ara330/Outfit_Aura/static/images/" + gender + "/" + type + "/" + category
    print ("Final selected directory:" + other_directory)
    # Example usage
    top_3_recommendations, inference_time = get_live_recommendations(selected_top_tensor, other_directory, model)
    print(f"Total inference time: {inference_time:.4f} seconds")

    results = []
    # Display the recommendations
    for i, (score, bottom_path) in enumerate(top_3_recommendations, start=1):
        print(f"Recommendation {i}: {bottom_path} with score {score:.4f}")
        bottom_path = bottom_path.replace("C:/Users/ARA/Desktop/ara330/Outfit_Aura", "")
        results.append(bottom_path)
        #bottom_image = Image.open(bottom_path)
        #bottom_image.show(title=f"Recommendation {i}")
    return results
# In[None]:

