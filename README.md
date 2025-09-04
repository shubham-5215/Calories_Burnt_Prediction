
## 🏃 Calories Burnt Prediction

This repository contains a machine learning project that predicts the number of calories a person burns during a workout. The project is implemented in a Jupyter Notebook, `calories-burnt-prediction.ipynb`, and uses a dataset from Kaggle.

---

### **Model and Methodology**

The primary model used is **Linear Regression**, which predicts calories burned based on several features. The notebook also explores other regression models, such as `DecisionTreeRegressor` and `RandomForestRegressor`. The linear regression model achieved **94% accuracy**.

The predictive features include:

* **Age**
* **Gender**
* **Height**
* **Weight**
* **Duration** (of the workout)
* **Heart Rate**
* **Body Temperature**

The project focuses on building a simple yet effective model for fitness tracking applications.

---

### **Dataset and Preprocessing**

The dataset, sourced from Kaggle, contains over 15,000 records of exercise and calorie data. Key steps in data preparation included:

* Merging two separate CSV files, `exercise.csv` and `calories.csv`, into a single dataframe.
* Handling categorical data by encoding the 'Gender' column.
* Data preprocessing, including **normalization** and **feature selection**, to optimize model performance.
* Data visualization was performed using libraries like `matplotlib` and `seaborn` to explore feature correlations.

---

### **Dependencies**

The project relies on standard Python libraries for data science and machine learning, including:

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `sklearn`

---
***
---

## ⚕️ Medical Image Segmentation with CE-Net

This repository presents an implementation of the CE-Net architecture, a deep learning model designed for 2D medical image segmentation. The focus of this project is on segmenting lungs from CT scan images, building upon the principles outlined in the **[CE-Net paper](https://arxiv.org/abs/1903.02740)**.

---

### **Model Breakdown**

The core of this project is the Context Encoder Network (CE-Net). The model is structured to capture both local and global features effectively:

* **Feature Encoder:** A ResNet-34 backbone is used to efficiently extract rich features from the input images.
* **Dense Atrous Convolution (DAC) Block:** This unique block utilizes dilated convolutions to expand the receptive field, allowing the model to incorporate a wider range of contextual information without increasing computational cost.
* **Residual Multi-kernel Pooling (RMP) Block:** This component is key to handling objects of varying sizes. It employs pooling operations with different kernel sizes to collect and combine features at multiple scales.
* **Feature Decoder:** The decoder network is responsible for reconstructing the final segmentation mask from the combined features.

For a visual representation of the CE-Net architecture, you can refer to the following diagram: [CE-Net Structure](https://pic3.zhimg.com/v2-c152b10827b18b63a79fce175a93a08e_1440w.jpg?source=172ae18b)

---

### **Dataset and Preprocessing**

The model was trained and evaluated on a publicly available dataset of 2D CT scans and their corresponding lung masks from Kaggle. The notebook handles all the necessary data preparation steps:

* **Image Resizing:** All images are resized to a standard `448x448` resolution.
* **Data Augmentation:** The dataset is augmented with techniques like random horizontal and vertical flips, as well as random contrast adjustments, to improve the model's robustness and generalization.

---

### **Getting Started**

#### **Prerequisites**

To run the notebook, you will need the following Python libraries:

* `os`
* `numpy`
* `pandas`
* `math`
* `cv2`
* `sklearn`
* `tensorflow`
* `matplotlib.pyplot`

The model itself is built and trained using the TensorFlow Keras API.

#### **Training and Evaluation**

The training process uses a combined Binary Cross-Entropy and Dice loss function (`bce_dice_loss`). The model is optimized using the Adam optimizer with an initial learning rate of `0.0001`.

Key aspects of the training workflow include:

* **Learning Rate Scheduler:** The learning rate is adjusted over time to facilitate more stable training.
* **Early Stopping:** To prevent overfitting, training stops if the validation Intersection over Union (IoU) metric does not show improvement for 15 consecutive epochs.
* **Model Checkpoint:** The best performing model weights (based on validation IoU) are saved, ensuring you retain the most accurate version of the model.

The model's performance is monitored using a set of standard metrics: Binary Cross-Entropy (`bce`), Dice Coefficient (`dice_coeff`), and Intersection over Union (`iou`). After training, the notebook provides a visual evaluation by displaying example predictions alongside the ground truth masks.
