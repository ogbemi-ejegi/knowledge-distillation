from utils import get_dataloaders

train_loader, test_loader = get_dataloaders()

images, labels = next(iter(train_loader))

print("Image Tensor Shape:", images.shape)
print("Label Tensor Shape:", labels.shape)

print("\nFirst 10 Labels:")
print(labels[:10])

print("\nFirst Image Tensor:")
print(images[0])