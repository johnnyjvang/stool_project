from matplotlib import pyplot as plt

img_path = 'sample_stool.jpg'
image = enhance_image(img_path)
segmented, mask = extract_stool_region(image)
augmented_tensor = augment_image(segmented)

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Enhanced")

plt.subplot(1, 3, 2)
plt.imshow(segmented)
plt.title("Segmented")

plt.subplot(1, 3, 3)
plt.imshow(augmented_tensor.permute(1, 2, 0))
plt.title("Augmented")
plt.show()
