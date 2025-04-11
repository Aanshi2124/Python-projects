# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_lfw_pairs
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neural_network import MLPClassifier
# import numpy as np
# import os, cv2
# def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
#     """Helper function to plot a gallery of portraits"""
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
#     plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
#     for i in range(n_row * n_col):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
#         plt.title(titles[i], size=12)
#         plt.xticks(())
#         plt.yticks(())

# dir_name = "dataset/faces/"
# y=[];x=[];target_names=[]
# person_id=0;h=w=300
# n_samples=0
# class_names=[]
# for Akshay in os.listdir("dataset/dataset/faces"):
#     dir_path = "dataset/faces/Akshay" 
#     class_names.append(Akshay)
#     for image_name in os.listdir("dataset/dataset/faces/Akshay"):
#         image_path = "dataset/dataset/faces/Akshay/face_3.jpg" 
#         img = cv2.imread("dataset/dataset/faces/Akshay/face_3.jpg")
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         resized_image = cv2.resize(gray, (h,w))
#         v=resized_image.flatten()
#         x.append(v)
#         n_samples=n_samples+1
#         y.append(person_id)
#         target_names.append(Akshay)
#     person_id=person_id+1

# y=np.array(y)
# x=np.array(x)
# target_names=np.array(target_names)
# n_features=x.shape[1]
# print(y.shape,x.shape,target_names.shape)
# print("Number of samples: ",n_samples)
# n_classes=target_names.shape[0]
# print("Total dataset size:")
# print("n_samples: %d" % n_samples)
# print("n_features: %d" % n_features)
# print("n_classes: %d" % n_classes)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
# n_components = 150
# print("Extracting the top %d eigenfaces from %d faces" % (n_components, x_test.shape[0]))
# pca=PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(x_train)
# eigenfaces=pca.components_.reshape((n_components, h, w))
# eigenface_titles=["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# plot_gallery(eigenfaces, eigenface_titles, h, w)
# plt.show()
# print("Projecting the input data on the eigenfaces orthonormal basis")
# x_train_pca=pca.transform(x_train)
# x_test_pca=pca.transform(x_test)
# print(x_train_pca.shape, x_test_pca.shape)
# lda=LinearDiscriminantAnalysis()
# lda.fit(x_train_pca, y_train)
# x_train_lda=lda.transform(x_train_pca)
# x_test_lda=lda.transform(x_test_pca)
# print("Project done...")

# clf = MLPClassifier(random_state=1, hidden_layer_sizes=(10,10),max_iter=1000, verbose=True).fit(x_train_lda, y_train)
# print("Model Weights:")
# model_info=[coef.shape for coef in clf.coefs_]
# print(model_info)

# y_pred=[]; y_prob=[]
# for test_face in x_test_lda:
#     prob=clf.predict_proba([test_face])[0]
#     class_id=np.where(prob==np.max(prob))[0][0]
#     y_pred.append(class_id)
#     y_prob.append(np.max(prob))
# y_pred=np.array(y_pred)
# prediction_titles=[]
# true_positive=0
# for i in range(y_pred.shape[0]):
#     true_name=class_names[y_test[i]]
#     pred_name=class_names[y_pred[i]]
#     result='pred: %s, pr:%s \ntrue: %s' %  (pred_name, str(y_prob[i])[0:3], true_name)
#     prediction_titles.append(result)
#     if true_name==pred_name:
#         true_positive=true_positive + 1
# print("Accuracy:",true_positive*100/y_pred.shape[0]) 
# plot_gallery(x_test, prediction_titles, h, w)
# plt.show()       




import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import cv2

# Plotting helper function
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(min(n_row * n_col, len(images))):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=10)
        plt.xticks(())
        plt.yticks(())

# Parameters
image_dir = "dataset/faces"
h, w = 100, 100  # Resize dimensions
x = []
y = []
class_names = []
person_id = 0

# Read images
for person_name in os.listdir(image_dir):
    person_dir = os.path.join(image_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    class_names.append(person_name)
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (h, w))
        x.append(resized_image.flatten())
        y.append(person_id)
    person_id += 1

# Convert to arrays
x = np.array(x)
y = np.array(y)
target_names = np.array(class_names)
n_samples, n_features = x.shape
n_classes = len(class_names)

print(f"Number of samples: {n_samples}")
print(f"Number of features: {n_features}")
print(f"Number of classes: {n_classes}")

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# PCA
n_components = 100
print(f"Extracting the top {n_components} eigenfaces")
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(x_train)
eigenfaces = pca.components_.reshape((n_components, h, w))
plot_gallery(eigenfaces, [f"eigenface {i}" for i in range(len(eigenfaces))], h, w)
plt.show()

# Transform using PCA
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_train_pca, y_train)
x_train_lda = lda.transform(x_train_pca)
x_test_lda = lda.transform(x_test_pca)

# MLP Classifier
clf = MLPClassifier(random_state=1, hidden_layer_sizes=(50, 30), max_iter=1000, verbose=True)
clf.fit(x_train_lda, y_train)
print("Model structure (layer shapes):", [coef.shape for coef in clf.coefs_])

# Prediction and evaluation
y_pred = []
y_prob = []

for test_face in x_test_lda:
    prob = clf.predict_proba([test_face])[0]
    class_id = np.argmax(prob)
    y_pred.append(class_id)
    y_prob.append(prob[class_id])

# Accuracy calculation
y_pred = np.array(y_pred)
true_positive = np.sum(y_pred == y_test)
accuracy = (true_positive / len(y_test)) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Prepare titles
prediction_titles = []
for i in range(len(y_test)):
    true_name = class_names[y_test[i]]
    pred_name = class_names[y_pred[i]]
    confidence = f"{y_prob[i]:.2f}"
    title = f"pred: {pred_name}, pr: {confidence}\ntrue: {true_name}"
    prediction_titles.append(title)

# Show prediction results
plot_gallery(x_test, prediction_titles, h, w)
plt.show()
