import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from leaf_net.model import EcoConvNet

# Define categories (e.g., CIFAR-10 classes)
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Note: In a real-world scenario, you would load your fine-tuned weights here using:
# model = EcoConvNet.from_pretrained("your-hf-username/ecoconvnet-cifar10")
model = EcoConvNet(img_size=(32, 32), num_classes=10)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def predict(image):
    if image is None:
        return None
    
    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Format the output for Gradio
    return {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="EcoConvNet Image Classifier",
    description="Upload an image to test the lightweight EcoConvNet model. Currently set up for CIFAR-10 format inputs.",
    examples=[
        # You can provide path strings to example images here
    ]
)

if __name__ == "__main__":
    demo.launch()
