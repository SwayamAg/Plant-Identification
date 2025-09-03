# 🌿 Plant Identification API

A FastAPI-based machine learning API that identifies plant species from images. It uses a TensorFlow/Keras CNN to classify images into 24 categories (derived from the `Data/` folders). The deployed model is ~300MB (`plant_species_Model_kaggle.h5`).

## 🌱 Supported Plant Types

The model can identify the following 24 plant types (derived from `Data/`):
- **Amla**
- **Arali**
- **Ashoka**
- **Ashwagandha**
- **Avacado**
- **Bamboo**
- **Basale**
- **Castor**
- **Corn**
- **Curry_Leaf**
- **Doddapatre**
- **Ganike**
- **Guava**
- **Henna**
- **Mint**
- **Nooni**
- **Pappaya**
- **Rose**
- **Wood_sorel**
- **aloevera**
- **banana**
- **mango**
- **orange**
- **watermelon**

## ✨ Features

- **FastAPI-based REST API** with automatic documentation
- **Real-time image classification** with confidence scores
- **CORS support** for web applications
- **Health check endpoint** for monitoring
- **Comprehensive error handling** and logging
- **Railway/Heroku/Render deployment ready** with Procfile
- **Local testing script** for quick predictions

## 🚀 Quick Start

### 🌐 **Live Demo**
Your API is live and ready to use!
- **API Base URL**: `https://your-api-base.example.com`
- **Interactive Documentation**: `https://your-api-base.example.com/docs`
- **Health Check**: `https://your-api-base.example.com/health`

### Prerequisites

- Python 3.10.12
- TensorFlow 2.10.0
- FastAPI
- Other dependencies (see `requirements.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SwayamAg/Plant-Identification
   cd Plants_Identification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the model file is present**
   - The API downloads `plant_species_Model_kaggle.h5` (~300MB) from Google Drive on first run
   - Alternatively, place `plant_species_Model_kaggle.h5` in the project root manually

### Running the API Locally

1. **Start the FastAPI server**
   ```bash
   python app.py
   ```

2. **Access the API**
   - API will be available at: `http://localhost:8000`
   - Interactive documentation: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

3. **Alternative: run using Uvicorn**
   ```bash
   # auto-reload on code changes (local only)
   uvicorn app:app --reload

   # specify host/port
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

## 📖 API Documentation

### Endpoints

#### 1. **GET /** - Root endpoint
Returns a welcome message.
```json
{
  "message": "Plant Type Classifier API is running!"
}
```

#### 2. **GET /health** - Health check
Returns the status of the API and model.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "plant_species_Model_kaggle.h5",
  "model_exists": true
}
```

#### 3. **POST /predict** - Image classification
Upload an image file to get plant classification.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "predicted_class": "mango",
  "confidence": 95.67
}
```

 

## 🧪 Local Testing

Use the included `test.py` script for quick local testing:

```bash
# Test with a specific image
python test.py path/to/your/image.jpg

# Or modify the IMAGE_PATH variable in test.py
```

## 🚀 Deployment

### Heroku Deployment

This project is configured for Heroku deployment:

1. **Create a Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Deploy to Heroku**
   ```bash
   git add .
   git commit -m "Deploy plant identification API"
   git push heroku main
   ```

3. **Scale the dyno**
   ```bash
   heroku ps:scale web=1
   ```

### Environment Variables

The API uses the following environment variables:
- `PORT`: Port number (automatically set by Heroku)
- `TF_CPP_MIN_LOG_LEVEL`: TensorFlow logging level (set to '3' to suppress logs)

## 📁 Project Structure

```
Plants_Identification/
├── app.py                           # FastAPI application with CORS support
├── Requirements.txt                 # Python dependencies and versions
├── Procfile                         # Deployment configuration
├── runtime.txt                      # Python runtime version
├── .gitignore                       # Git ignore rules
├── readme.md                        # Project documentation
└── plant-cnn-model_kaggle.ipynb     # Notebook (training/experiments)
```

## 🔧 Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input Size**: 224x224 pixels
- **Classes**: 24 plant types
- **Model Size**: ~300MB
- **Training**: Custom dataset with train/validation/test splits

## 🛠️ Development

### Adding New Plant Types

1. Add a new class folder under `Data/`
2. Retrain the model with the updated dataset
3. Deploy the updated model file (`plant_species_Model_kaggle.h5`)

### Model Training

The training process is documented in `Plant_Classification_train.ipynb`. To retrain:

1. Prepare your dataset in the `train/`, `val/`, and `test/` directories
2. Run the Jupyter notebook
3. Replace `plant_species_Model_kaggle.h5` with the new model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- FastAPI team for the excellent web framework
- The plant dataset contributors

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs` when running locally
- Review the health endpoint at `/health` for system status

---

## 👨‍💻 Made by Swayam

- **LinkedIn**: [Swayam Agarwal](https://www.linkedin.com/in/swayam-agarwal/)
- **GitHub**: [SwayamAg](https://github.com/SwayamAg)

---

**Note**: The API auto-downloads `plant_species_Model_kaggle.h5` on first run; you can also place the ~300MB model in the project root manually.
