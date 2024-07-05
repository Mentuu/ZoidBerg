import os
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
import cv2
import joblib

param_grid = {
    'svc__C': [0.1, 1, 10, 100], #0.1, 1, 10, 100
    'svc__gamma': [1, 0.1, 0.01, 0.001], #1, 0.1, 0.01, 0.001
    'svc__kernel': ['rbf'] #'linear', 'rbf', 'poly', 'sigmoid', 'precomputed', 'callable', 'auto'
}

path_train_malades = 'C:/Users/Mentu/Documents/IA/Zoidberg/train/PNEUMONIA'
path_train_non_malades = 'C:/Users/Mentu/Documents/IA/Zoidberg/train/NORMAL'
path_test_malades = 'C:/Users/Mentu/Documents/IA/Zoidberg/test/PNEUMONIA'
path_test_non_malades = 'C:/Users/Mentu/Documents/IA/Zoidberg/test/NORMAL'

def timing_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper 

@timing_function
def load_images_from_two_folders(folder1, folder2, image_size=(64, 64)):
    images, labels = [], []

    # Load images from the first folder
    for filename in os.listdir(folder1):
        if filename.endswith('.jpeg'): # Check file extension if necessary
            img = Image.open(os.path.join(folder1, filename)).convert('L')  # Convert to grayscale
            img = crop_lungs_from_image(img)
            img = img.resize(image_size)
            images.append(np.array(img).flatten())  # Flatten the image
            labels.append(1)

    # Load images from the second folder
    for filename in os.listdir(folder2):
        if filename.endswith('.jpeg'):  # Check file extension if necessary
            img = Image.open(os.path.join(folder2, filename)).convert('L')  # Convert to grayscale
            img = crop_lungs_from_image(img)
            img = img.resize(image_size)
            images.append(np.array(img).flatten())  # Flatten the image
            labels.append(0)

    return np.array(images), np.array(labels)

def crop_lungs_from_image(img):
    """Crops only lungs image from image.
    :param img: image to be cropped.
    :returns: cropped image."""
    img_cv = np.array(img)
    _, thresh = cv2.threshold(img_cv, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img_cv = img_cv[y:y + h, x:x + w]
    cropped_img = Image.fromarray(cropped_img_cv)
    return cropped_img

# Load images and labels
train_images, train_labels = load_images_from_two_folders(path_train_malades, path_train_non_malades)
# Split the data into training and validation sets (85% train, 15% validation)
# train_images, val_images, train_labels, val_labels = train_test_split(
#     train_images, train_labels, test_size=0.15, random_state=42, stratify=train_labels    
# )

# load test images and labels
test_images, test_labels = load_images_from_two_folders(path_test_malades, path_test_non_malades)

# Création d'un pipeline avec mise à l'échelle des données et SVM
pipeline = make_pipeline(StandardScaler(), SVC(probability=True, class_weight='balanced', verbose=True, kernel='rbf', C=1))

#entrainement du modele
pipeline.fit(train_images, train_labels)

# grid_search = GridSearchCV(pipeline, param_grid, refit=True, verbose=2, cv=5) 
# best parameters
# grid_search.fit(val_images, val_labels)

# print(f"Best parameters found: {grid_search.best_params_}")

# best_model = GridSearchCV.best_estimator_

# Prediction
predicted_labels = pipeline.predict(test_images)

# Classification report
print(classification_report(test_labels, predicted_labels))

#confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)
#i want something more detailed with plt
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
# Set ticks and tick labels
ax.set(xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=['Normal', 'Pneumonia'],
    yticklabels=['Normal', 'Pneumonia'],
    title='Confusion Matrix',
    ylabel='True label',
    xlabel='Predicted label')
# Loop over data dimensions and create text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
        ha="center", va="center",
        color="white" if cm[i, j] > thresh else "black")
plt.show()




# Compute ROC curve and ROC area for the test set
probabilities = pipeline.predict_proba(test_images)[:, 1]
fpr, tpr, _ = roc_curve(test_labels, probabilities)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Save the model
joblib.dump(pipeline, 'trained_model.pkl')