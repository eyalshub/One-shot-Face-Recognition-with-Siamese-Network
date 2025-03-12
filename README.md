# One-shot Face Recognition with Siamese Network

## 📂 Project Overview
This project is focused on implementing a Siamese network for one-shot face recognition. The model is trained on pairs of images and predicts whether the two images represent the same person. The dataset consists of 5,749 individuals with over 12,000 images, and for training, we used 1,100 identical (Class 1) pairs and 1,100 non-identical (Class 0) pairs.

### Dataset Overview
- **Total Individuals:** 5,749
- **Training Pairs:** 1,100 identical (Class 1) pairs and 1,100 non-identical (Class 0) pairs
- **Test Pairs:** 500 pairs for each class
- **Validation Pairs:** 20% of training pairs (440 pairs) set aside for validation
- **Image Size:** All images resized to 105x105 pixels with 1 channel (grayscale)

## 🧠 Architecture Description
The Siamese network consists of two identical sub-networks that process two input images to compute an embedding for each. The absolute difference between the two embeddings is calculated to measure the similarity between the images.

### Layer-by-Layer Breakdown

| Layer | Input Size    | Filters/Units | Kernel Size | Maxpooling | Activation Function | Output Size     |
|-------|---------------|---------------|-------------|------------|---------------------|-----------------|
| 1     | (105, 105, 1) | 64            | (10, 10)     | (2, 2)     | ReLU                | (48, 48, 64)    |
| 2     | (48, 48, 64)  | 128           | (7, 7)       | (2, 2)     | ReLU                | (21, 21, 128)   |
| 3     | (21, 21, 128) | 128           | (4, 4)       | (2, 2)     | ReLU                | (9, 9, 128)     |
| 4     | (9, 9, 128)   | 256           | (4, 4)       | None       | ReLU                | (6, 6, 256)     |
| 5     | (4096,)       | 1             | -           | -          | Sigmoid             | (1,)            |

### Key Features:
- **Optimizer:** Adam optimizer
- **Loss Function:** Binary cross-entropy
- **Regularization:** L2 regularization
- **Metric:** Accuracy
- **Early Stopping:** Enabled to stop training if validation accuracy doesn’t improve for 5 epochs

## 🔧 Hyperparameters and Experimentation
We conducted experiments with different combinations of hyperparameters to find the optimal configuration for the model:

- **Dropout Rate:** 0, 0.25, 0.5
- **Learning Rate:** 0.05, 0.005, 0.0005
- **Batch Size:** 16, 32, 64
- **Early Stopping:** True, False

Each experiment was run for up to 20 epochs with early stopping based on validation accuracy. The following observations were made:
- **Best Dropout Rate:** 0.25
- **Best Learning Rate:** 0.0005
- **Best Batch Size:** 64 (despite minimal differences across batch sizes)
- **Early Stopping:** No significant effect

### Final Configuration:
- **Dropout Rate:** 0.25
- **Learning Rate:** 0.0005
- **Batch Size:** 64
- **Early Stopping:** Enabled

## 🏆 Results
The network achieved good performance in one-shot face recognition. Some key observations:
- The network performed well when the two images were similar, such as when they are of the same person.
- Confusion arose in cases where the images had similar features (e.g., similar hairstyle or facial expression), even though they represented different individuals.
- The model had difficulty correctly classifying images where multiple people were present or when the facial expressions were very different.

### Example Scenarios:
1. **Same Person:** The model correctly identified two images of the same person, with similar facial features and hairstyle.
2. **Different Person (similar features):** The model misclassified the images of two different people with similar features, likely due to facial expressions or hairstyle similarities.
3. **Confusion due to Multiple People:** In one image, an additional person entered the frame, confusing the model, as it used features from both individuals to make a prediction.

## 🎯 Future Work and Improvements:
- **Enhance Generalization:** Experiment with more advanced regularization techniques such as **Dropout** and **Data Augmentation**.
- **Optimize Model Hyperparameters:** Further tuning of hyperparameters could yield slightly improved performance.
- **Explore Other Architectures:** Investigate alternative architectures like **triplet loss networks** or **contrastive loss networks** for better embedding learning.

