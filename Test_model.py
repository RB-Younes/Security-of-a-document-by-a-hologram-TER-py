from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

test_folder = 'E:/dataset MIDV HOLO/Mosaics splited/test'


test_datagen = ImageDataGenerator()  # Image normalization.

test_set = test_datagen.flow_from_directory(test_folder,
                                            target_size=(224, 224),
                                            class_mode='binary',
                                            batch_size=1,
                                            shuffle=False)

"""# **Performance Metrics**"""

# load best 64b model
model = tf.keras.models.load_model('best-without-weights/last_epoch_model.h5')

model.summary()

# load the saved model from file
with open('best-without-weights/history.pkl', 'rb') as file:
    hist = pickle.load(file)

# access the history attribute of the model
index_of_min = np.argmin(hist['val_loss'])  # min val loss
loss = hist['loss'][index_of_min]
val_loss = hist['val_loss'][index_of_min]
acc = hist['acc'][index_of_min]
val_acc = hist['val_acc'][index_of_min]
print('---------------------best_model-----------------------')
print('acc: ' + "{:0.4f}".format(acc))
print('loss: ' + "{:0.4f}".format(loss))
print('val_acc: ' + "{:0.4f}".format(val_acc))
print('val_loss: ' + "{:0.4f}".format(val_loss))

index_last_epoch = len(hist['loss']) - 1
loss = hist['loss'][index_last_epoch]
val_loss = hist['val_loss'][index_last_epoch]
acc = hist['acc'][index_last_epoch]
val_acc = hist['val_acc'][index_last_epoch]

print('---------------------last_epoch_model-----------------------')
print('acc: ' + "{:0.4f}".format(acc))
print('loss: ' + "{:0.4f}".format(loss))
print('val_acc: ' + "{:0.4f}".format(val_acc))
print('val_loss: ' + "{:0.4f}".format(val_loss))

"""

*   Test evaluation

"""

# test evaluation
test_loss, test_acc = model.evaluate(test_set)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

"""

*   Confusion matrix

"""

test_set.reset()
Y_pred = model.predict(test_set)
y_pred_prob = Y_pred.squeeze()  # Ensure it's a 1D array
y_pred = np.where(y_pred_prob >= 0.5, 1, 0)

confusion_matrix_res = confusion_matrix(test_set.classes, y_pred)

sns.heatmap(confusion_matrix_res, annot=True, fmt="d")
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.savefig('best-without-weights/confusion_matrix_res.png')
plt.show()

"""

*   precision  -  recall - f1-score - support

"""

print(classification_report(test_set.classes, y_pred, labels=[0, 1], digits=4))

"""

*  ROC Curve

"""
# Calculate sensitivity and specificity for each class
sensitivity_list = []
specificity_list = []
# Calculate sensitivity and specificity for each class
for i in range(2):
    TP = confusion_matrix_res[i, i]
    TN = np.sum(confusion_matrix_res) - np.sum(confusion_matrix_res[i, :]) - np.sum(confusion_matrix_res[:, i]) + \
         confusion_matrix_res[
             i, i]
    FP = np.sum(confusion_matrix_res[:, i]) - confusion_matrix_res[i, i]
    FN = np.sum(confusion_matrix_res[i, :]) - confusion_matrix_res[i, i]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)

    print(f"Class {i}: Sensitivity = {sensitivity:.4f}, Specificity = {specificity:.4f}")

# Calculate macro-averaged sensitivity and specificity
macro_sensitivity = np.mean(sensitivity_list)
macro_specificity = np.mean(specificity_list)
print(f"Macro-averaged Sensitivity = {macro_sensitivity:.4f}, Macro-averaged Specificity = {macro_specificity:.4f}")

from sklearn.preprocessing import label_binarize

# Binarize true labels for multi-class classification
y_true_binary = label_binarize(test_set.classes, classes=[0, 1, 2])
y_pred = label_binarize(y_pred, classes=[0, 1, 2])
print(y_true_binary.shape)
print(y_pred.shape)
# Calculate ROC curve and AUC for each class
# Calculate ROC curve and AUC for each class
n_classes = 3  # replace with the actual number of classes in your problem
fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute ROC curve and AUC for each class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binary.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binary.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and AUC
# For macro-average, we average the ROC curve across all classes
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 3
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot ROC curve for each class and micro/macro average
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--')
for i in range(3):
    ax.plot(fpr[i], tpr[i], label=f"Class {i}, AUC = {roc_auc[i]:.4f}")
ax.plot(fpr["micro"], tpr["micro"], label="micro-average, AUC = {:.4f}".format(roc_auc["micro"]), linestyle=':',
        linewidth=4)
ax.plot(fpr["macro"], tpr["macro"], label="macro-average, AUC = {:.4f}".format(roc_auc["macro"]), linestyle=':',
        linewidth=4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
plt.savefig('best-without-weights/ROC curve.png')
plt.show()
