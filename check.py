from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import cv2

model = YOLO(r'runs\segment\train\weights\best.pt')

img = cv2.imread(r'photo\102.jpg')

results = model(img, imgsz=640, iou=0.4, conf=0.7)

classes = results[0].boxes.cls.cpu().numpy()
class_names = results[0].names

masks = results[0].masks.data
num_masks = masks.shape[0]

colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]

masks_overlay = np.zeros_like(img)

labeled_image = img.copy()

for i in range(num_masks):
    color = colors[i]
    mask = masks[i].cpu()

    mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    class_index = int(classes[i])
    class_name = class_names[class_index]

    mask_contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(labeled_image, mask_contours, -1, color, 2)
    cv2.putText(labeled_image, class_name, (int(mask_contours[0][:, 0, 0].mean()), (int(mask_contours[0][:, 0, 1].mean()))), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

plt.figure(figsize=(8, 8), dpi=150)
labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
plt.imshow(labeled_image)
plt.axis('off')
plt.show()