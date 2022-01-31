import matplotlib.pyplot as plt

def show_images(train_ds, class_names, n = 9):
    for images, labels in train_ds:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")