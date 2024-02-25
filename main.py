from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()

# Define the path to the model file
model_filename = os.path.join(os.path.dirname(__file__), 'model', 'dog_emotion_model.h5')

# Load the trained model
model = load_model(model_filename)

# Define emotion labels (customize based on your model)
emotion_labels = ["Happy", "Angry", "Relaxed", "Sad"]

def preprocess_image(img):
    img = image.load_img(img, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

def predict_dog_emotion(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)[0]  # Assuming a single prediction
    predicted_class = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_class]

    # Additional details about the prediction
    confidence_scores = {label: round(float(score), 4) for label, score in zip(emotion_labels, prediction)}

    return {"emotion": predicted_emotion, "analysis": confidence_scores}

@app.post("/predict-dog-emotion", response_model=dict)
async def predict_dog_emotion_endpoint(file: UploadFile = File(...)):
    try:
        # Check if the uploaded file is an image
        if file.content_type.startswith("image/"):
            # Read the image content
            contents = await file.read()

            # Make prediction
            prediction_details = predict_dog_emotion(io.BytesIO(contents))

            return JSONResponse(content=prediction_details)

        else:
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
