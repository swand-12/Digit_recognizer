import streamlit as st
import torch
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Load the trained model
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):  # in_channels = 1 because grayscale
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)  # Batch Normalization
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)  # Increased dropout rate

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 256)  # Increased number of neurons
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.shape[0], -1)  # Flatten the output

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# Load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)  # Move the model to the appropriate device
model.load_state_dict(torch.load("digit_recognizer_model.pth", map_location=device))  # Load model weights
model.eval()  # Set the model to evaluation mode

# Dictionary for digit to word mapping
digit_to_word = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

# Streamlit interface
st.title("Digit Recognizer")
st.write("Draw a digit in the box below and click 'Predict'.")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Feedback data storage
feedback_data = []

# Button to predict
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Process the drawn image
        img = canvas_result.image_data[:, :, :3]
        img = Image.fromarray(img)
        img = img.convert("L")  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to MNIST size
        img = np.array(img) / 255.0  # Normalize
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions

        # Ensure the input tensor has the correct shape
        if img.shape != (1, 1, 28, 28):
            st.error("Input shape is incorrect. Expected shape is (1, 1, 28, 28). Current shape: " + str(img.shape))
        else:
            # Make a prediction
            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)
                predicted_digit = predicted.item()
                
                # Display predicted digit and word equivalent
                st.write(f"Predicted Digit: {predicted_digit} ({digit_to_word[predicted_digit]})")

                # Ask for user confirmation
                correct_answer = st.number_input("Is this correct? Enter the correct digit if not:", min_value=0, max_value=9, value=predicted_digit)
                
                # If the prediction is wrong, save the feedback
                if predicted_digit != correct_answer:
                    feedback_data.append({
                        'image': img.squeeze().cpu().numpy(),
                        'predicted': predicted_digit,
                        'correct': correct_answer
                    })
                    st.write("Thanks for your feedback! The correct digit has been recorded.")

# Save feedback data to a CSV file
if st.button("Save Feedback"):
    if feedback_data:
        feedback_df = pd.DataFrame(feedback_data)
        feedback_df.to_csv("feedback_data.csv", index=False)
        st.write("Feedback saved to feedback_data.csv")
    else:
        st.write("No feedback to save.")
