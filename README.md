
#            AI-Driven Assistant for Rapid Dermatology Triage


The AI-Driven Assistant utilizes deep learning to analyze skin images, reducing dermatology care delays by assessing case urgency and providing recommendations for faster access to treatment. By automating initial analysis and continuously updating its knowledge, the system ensures precise evaluations across various skin conditions and types.

#            Problem Statement
To create an AI-driven tool that rapidly analyzes skin images, assesses case urgency, and provides care recommendations, enhancing dermatology service efficiency and patient access.

#            Proposed Solution
The proposed system is an AI assistant that analyzes skin images to classify conditions and recommend triage priority.
It uses a fine-tuned model for accurate identification, enabling faster and prioritized care.
Regular updates maintain accuracy across diverse cases.


#            Methodology
Data Collection: Gather diverse dermatology images.
Preprocessing: Enhance images and verify labels.
Model Training: Fine-tune a pre-trained model (e.g., ResNet).
Classification: Identify skin conditions from features.
Triage Recommendation: Assign urgency levels based on classification

#             Implementation
Base Algorithm: Convolutional Neural Networks (CNNs) for image analysis.
Model Used: ResNet50 architecture with residual learning.
Technique: Transfer learning with a pre-trained ResNet50 model on ImageNet.
Optimization: Used Adam/SGD optimizer for model training.
Loss Function: Applied Categorical Crossentropy for classification.

# Outputs
<img width="657" height="353" alt="image" src="https://github.com/user-attachments/assets/5d4e8476-e7a1-4b3b-af80-f50faf949190" />
                                          Login Screen
<img width="604" height="374" alt="image" src="https://github.com/user-attachments/assets/be27804d-b50c-4c2d-9fa9-0bb2d16f5c6e" />
<img width="686" height="443" alt="image" src="https://github.com/user-attachments/assets/b7eeba57-a319-4bf9-88bd-19981ffa6b11" />
                                          Predicts Disease name & its Severity
<img width="608" height="431" alt="image" src="https://github.com/user-attachments/assets/5d4d5e6f-69d2-463a-a7d8-8052b08eb62c" />
                                          Precautions
