import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
import pandas as pd
import joblib

# Load environment variables
load_dotenv()

# Create a chat model
chat = ChatOpenAI()

# Create a prompt for the chat model
prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

# Initialize a language model chain with the chat model and prompt
chain = LLMChain(
    llm=chat,
    prompt=prompt,
)

# Load the InceptionV3 model pre-trained on ImageNet data
model = InceptionV3(weights='imagenet')

# Function to process the image and make predictions
def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))  # InceptionV3 input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict classes using the pre-trained model
    preds = model.predict(img_array)
    predictions = decode_predictions(preds, top=3)[0]  # Decode predictions, get top 3 classes

    # Extract the class with the highest probability
    top_prediction = predictions[0]

    # Print predictions
    print("Predictions for", image_path)
    for pred in predictions:
        print(f"{pred[1]}: {pred[2]*100:.2f}%")

    return top_prediction[1]  # Return the class label with the highest probability

# Example usage:
image_path = '1.jpeg'
predicted_class = predict_image(image_path)

# Set up values for envelope and pillow based on the predicted class
if predicted_class.lower() == 'envelope':
    envelope = True
    pillow = False
elif predicted_class.lower() == 'pillow':
    envelope = False
    pillow = True
else:
    envelope = False
    pillow = False

# Generate content for the chat based on the predicted image class
content = f"Write a Instagram caption for an image of an {predicted_class}"

# Get response from the language model chain
result = chain({"content": content})

# Print the generated caption
print("caption:", result["text"])

# Load the regression models for predicting likes, shares, and saves
loaded_model_likes = joblib.load('model_likes.pkl')
loaded_model_shares = joblib.load('model_shares.pkl')
loaded_model_saves = joblib.load('model_saves.pkl')

# Example of using the loaded models for prediction
# Assuming Followers=1000, Time=10, envelope, and pillow set based on prediction
Followers = int(input("Enter number of followers:"))
Time = int(input("Enter Time to post:"))

# Create input data for the regression models
input_data = pd.DataFrame([[Followers, Time, envelope, pillow]], columns=['Followers', 'Time', 'envelope', 'pillow'])

# Predict likes, shares, and saves
predicted_likes = int(loaded_model_likes.predict(input_data)[0])
predicted_shares = int(loaded_model_shares.predict(input_data)[0])
predicted_saves = int(loaded_model_saves.predict(input_data)[0])

# Print prediction results
print("-------------------------------------------OUTPUT------------------------------------------------------")
print("caption:", result["text"])
print("Predicted likes:", predicted_likes)
print("Predicted shares:", predicted_shares)
print("Predicted saves:", predicted_saves)
