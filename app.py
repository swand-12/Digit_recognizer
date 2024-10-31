import streamlit as st
import torch
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os

# CSS for custom styles
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff; /* Light blue background */
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50; /* Green button color */
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .predicted-text {
        color: #2E86C1; /* Stylish blue color for predicted text */
        font-size: 24px;
        font-weight: bold;
    }
    .canvas-title {
        color: #34495E; /* Dark blue for canvas title */
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("digit_recognizer_model.pth", map_location=device))
model.eval()

# Digit to word mapping
digit_to_word = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

# Streamlit interface
st.title("🖌️ Digit Recognizer")
st.markdown('<div class="canvas-title">Draw a digit in the box below and click "Predict".</div>', unsafe_allow_html=True)

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

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Process the drawn image
        img = canvas_result.image_data[:, :, :3]
        img = Image.fromarray(img)
        img = img.convert("L")
        img = img.resize((28, 28))
        img = np.array(img) / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

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
                st.markdown(f'<div class="predicted-text">Predicted Digit: {predicted_digit} ({digit_to_word[predicted_digit]})</div>', unsafe_allow_html=True)

                # Ask for user confirmation
                is_correct = st.radio("Is this prediction correct?", ('Yes', 'No'))
                
                if is_correct == 'No':
                    correct_answer = st.number_input("Enter the correct digit:", min_value=0, max_value=9, value=predicted_digit)
                    feedback_data.append({
                        'image': img.squeeze().cpu().numpy(),
                        'predicted': predicted_digit,
                        'correct': correct_answer
                    })
                    st.write("Thanks for your feedback! The correct digit has been recorded.")

# Save feedback data to a CSV file with append functionality
if st.button("Save Feedback"):
    if feedback_data:  # Only save if feedback_data has items
        feedback_df = pd.DataFrame(feedback_data)

        # Check if file exists; if so, append without headers
        if os.path.isfile("feedback_data.csv"):
            feedback_df.to_csv("feedback_data.csv", mode='a', header=False, index=False)
        else:
            feedback_df.to_csv("feedback_data.csv", index=False)  # Write with headers if new file

        st.write("Feedback saved to feedback_data.csv")

        # Clear feedback_data after saving to avoid duplicates
        feedback_data.clear()
    else:
        st.warning("No new feedback to save.")
