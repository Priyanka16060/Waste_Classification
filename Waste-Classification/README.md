Waste Classification Using MobileNetV2 (Computer Vision+Data Analysis Project)

Efficient waste segregation is critical for sustainable waste management systems. This project presents an end-to-end computer vision solution that classifies waste images into multiple categories using deep learning and transfer learning. The solution is designed to be scalable, reusable, and analytics-driven, bridging machine learning with data visualization.

Manual waste segregation is time-consuming, error-prone, and inefficient. Automating this process using computer vision can:

Improve recycling efficiency

Reduce environmental impact

Support smart waste management systems

The objective of this project is to accurately classify waste images into predefined categories using a pretrained deep learning model.

Model & Learning Strategy

Used MobileNetV2, a lightweight yet powerful CNN architecture

Applied transfer learning using ImageNet pretrained weights

Fine-tuned upper layers to adapt the model to domain-specific waste images

Data Pipeline

Image preprocessing and resizing to 224×224

Data augmentation (rotation, zoom, shifts, flips) to improve generalization

Stratified train-validation split to avoid data leakage

Training & Optimization



Two-phase training strategy:

Feature extraction with frozen base layers

Fine-tuning with selective unfreezing

Optimized using Adam optimizer with learning-rate scheduling

Prevented overfitting using early stopping and dropout

Evaluation & Results

Achieved >90% validation accuracy

Evaluated performance using:

Validation accuracy & loss curves

Confusion matrix

Generated prediction outputs and exported results for analytical visualization

Model predictions were further analyzed using an interactive Tableau dashboard to identify:

Class distribution

Correct vs incorrect predictions

Misclassification patterns


Programming: Python

Deep Learning: TensorFlow, Keras

Model Architecture: MobileNetV2

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Tableau Public

Environment: Google Colab


Project Structure
Waste-Classification/
│
├── model/
│   └── waste_classification_mobilenetv2.h5
│
├── notebook.ipynb
│
├── waste_predictions.csv
│
├── dashboard_link.txt
│
└── README.md

Folder Details

model/: Saved trained model for reuse, inference, or deployment

notebook.ipynb: Complete experimentation and training pipeline

waste_predictions.csv: Structured prediction results for analytics

dashboard_link.txt: Public Tableau dashboard link

README.md: Project documentation

Dashboard & Insights

The prediction results were visualized using an interactive dashboard to:

Analyze dataset balance

Measure model reliability

Identify confusion between classes

Dashboard link is provided in dashboard_link.txt.

Future Enhancements

Deploy model as a real-time web application using Streamlit or Flask

Extend to real-world waste images with varied backgrounds

Integrate with IoT-based smart bin systems

Experiment with advanced architectures (EfficientNet, Vision Transformers)

Key Takeaways

Demonstrates a production-ready ML workflow

Combines computer vision, analytics, and storytelling

Designed with reusability and scalability in mind

Dashboard
See dashboard_link.txt for interactive visualization.

Future Improvements
- Deploy using Streamlit
- Improve accuracy with more data

Author

Developed as an independent AI/ML project to demonstrate applied skills in computer vision, deep learning, and data visualization.


