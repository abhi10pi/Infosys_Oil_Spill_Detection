# AI SpillGuard - FastAPI Oil Spill Detection System

## 🌊 Overview
AI SpillGuard is an advanced oil spill detection system powered by deep learning and FastAPI. It provides real-time analysis of satellite imagery to detect and monitor oil spills with high accuracy.

## ✨ Features

### Core Functionality
- **AI-Powered Detection**: Uses U-Net deep learning model for accurate oil spill segmentation
- **Real-time Processing**: Fast image analysis with immediate results
- **Interactive Web Interface**: Modern, responsive UI with drag-and-drop functionality
- **Comprehensive Analytics**: Detailed metrics including coverage percentage, severity assessment
- **Visual Results**: Side-by-side comparison of original, mask, and overlay images

### Advanced Features
- **Detection History**: Track and review past detections
- **System Statistics**: Real-time performance monitoring
- **Report Generation**: Automated report creation with environmental impact assessment
- **RESTful API**: Complete API endpoints for integration
- **Alert Configuration**: Customizable alert thresholds
- **Batch Processing**: Support for multiple image analysis
- **Export Functionality**: Download results and reports

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Trained U-Net model (`Unet_OilSpill.keras`)
- Required dependencies

### Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements_fastapi.txt
   ```

2. **Ensure Model File**
   Make sure `Unet_OilSpill.keras` is in the project root directory

3. **Run the Server**
   ```bash
   python run_server.py
   ```

4. **Access the Application**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative API Docs: http://localhost:8000/redoc

## 📁 Project Structure

```
Infy_Spng_Int_Proj/
├── main.py                 # Main FastAPI application
├── api_endpoints.py        # Additional API endpoints
├── run_server.py          # Server startup script
├── requirements_fastapi.txt # Dependencies
├── Unet_OilSpill.keras    # Trained model
├── static/                # Web assets
│   ├── index.html         # Main web interface
│   ├── style.css          # Styling
│   └── script.js          # JavaScript functionality
├── results/               # Detection results (auto-created)
└── README_FastAPI.md      # This file
```

## 🔧 API Endpoints

### Main Endpoints
- `GET /` - Web interface
- `POST /detect` - Oil spill detection
- `GET /history` - Detection history
- `GET /stats` - System statistics

### API v1 Endpoints
- `GET /api/v1/health` - System health check
- `GET /api/v1/detections` - All detections with pagination
- `GET /api/v1/detections/{id}` - Specific detection details
- `DELETE /api/v1/detections/{id}` - Delete detection
- `GET /api/v1/analytics/summary` - Analytics summary
- `POST /api/v1/alerts/configure` - Configure alerts
- `GET /api/v1/alerts/config` - Get alert configuration

## 💡 Usage Examples

### Web Interface
1. Open http://localhost:8000
2. Upload satellite image via drag-and-drop or file browser
3. View real-time analysis results
4. Download results or generate reports
5. Check detection history and system statistics

### API Usage
```python
import requests

# Upload image for detection
with open('satellite_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f}
    )
    result = response.json()

# Get system health
health = requests.get('http://localhost:8000/api/v1/health').json()

# Get detection history
history = requests.get('http://localhost:8000/api/v1/detections').json()
```

## 📊 Features Breakdown

### Detection Metrics
- **Coverage Percentage**: Percentage of image affected by oil spill
- **Severity Level**: High/Medium/Low based on coverage
- **Affected Pixels**: Number of pixels identified as oil spill
- **Processing Time**: Time taken for analysis

### Visual Analysis
- **Original Image**: Uploaded satellite image
- **Detection Mask**: Binary mask showing oil spill areas
- **Overlay Result**: Original image with highlighted spill areas

### System Monitoring
- **Total Detections**: Number of images processed
- **Average Coverage**: Mean coverage across all detections
- **Model Status**: Current model operational status
- **Last Updated**: Timestamp of last system update

## 🎨 UI Features

### Modern Design
- Responsive Bootstrap-based interface
- Gradient backgrounds and smooth animations
- Interactive cards and hover effects
- Professional color scheme

### User Experience
- Drag-and-drop file upload
- Real-time loading indicators
- Smooth scrolling navigation
- Mobile-responsive design
- Toast notifications for user feedback

## 🔒 Security Features
- File type validation
- Error handling and logging
- CORS middleware configuration
- Input sanitization

## 🚀 Deployment Options

### Local Development
```bash
python run_server.py
```

### Production Deployment
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_fastapi.txt .
RUN pip install -r requirements_fastapi.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 Performance Optimization
- Model caching with `@cache_resource`
- Efficient image processing
- Asynchronous request handling
- Static file serving optimization

## 🛠️ Customization

### Model Configuration
Update `MODEL_PATH` in `main.py` to use different models:
```python
MODEL_PATH = "path/to/your/model.keras"
```

### UI Customization
Modify `static/style.css` for custom styling:
```css
:root {
    --primary-color: #your-color;
    --secondary-color: #your-secondary-color;
}
```

### Alert Thresholds
Configure alert thresholds via API:
```python
requests.post('http://localhost:8000/api/v1/alerts/configure', json={
    "threshold": 10.0,
    "email": "admin@example.com",
    "enabled": True
})
```

## 🐛 Troubleshooting

### Common Issues
1. **Model Loading Error**: Ensure `Unet_OilSpill.keras` exists
2. **Port Already in Use**: Change port in `run_server.py`
3. **Memory Issues**: Reduce image size or batch processing
4. **CORS Errors**: Check CORS middleware configuration

### Logging
Check console output for detailed error messages and system status.

## 📝 License
This project is part of the Infosys Internship Program 2024.

## 🤝 Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📞 Support
For technical support or questions about the oil spill detection system, please refer to the project documentation or contact the development team.

---

**AI SpillGuard** - Protecting marine ecosystems through advanced AI technology 🌊
