import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 784)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x.view(-1, 1, 28, 28)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

z_dim = 100
batch_size = 64
epochs = 20
lr = 0.0002

transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))
])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

generator = Generator(z_dim=z_dim)
discriminator = Discriminator()
classifier_model = Classifier() 

criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
classifier_optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)
classification_criterion = nn.CrossEntropyLoss()


print("Training Classifier Model...")
classifier_model.train()
for epoch in range(20):  # Short training for demonstration; increase epochs for better accuracy
    for images, labels in train_loader:
        classifier_optimizer.zero_grad()
        outputs = classifier_model(images)
        loss = classification_criterion(outputs, labels)
        loss.backward()
        classifier_optimizer.step()
    print(f"Classifier Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

for epoch in range(20):
    for real_data, _ in train_loader:
        current_batch_size = real_data.size(0)

        # Train Discriminator
        disc_optimizer.zero_grad()
        real_data = real_data.view(current_batch_size, -1)

        real_labels = torch.ones(current_batch_size, 1)
        fake_labels = torch.zeros(current_batch_size, 1)

        outputs = discriminator(real_data)
        real_loss = criterion(outputs, real_labels)
        real_loss.backward()

        z = torch.randn(current_batch_size, z_dim)
        fake_data = generator(z)
        outputs = discriminator(fake_data.detach())
        fake_loss = criterion(outputs, fake_labels)
        fake_loss.backward()
        disc_optimizer.step()

        # Train Generator
        gen_optimizer.zero_grad()
        outputs = discriminator(fake_data)
        gen_loss = criterion(outputs, real_labels)
        gen_loss.backward()
        gen_optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], d_loss: {real_loss + fake_loss:.4f}, g_loss: {gen_loss:.4f}")
    
def robustness_test(classifier_model, generator, z_dim, num_samples=100):
    generator.eval()
    classifier_model.eval()
    correct = 0
    total = num_samples
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, z_dim)
            adversarial_example = generator(z)
            adversarial_example = adversarial_example.view(-1, 1, 28, 28)
            output = classifier_model(adversarial_example)
            pred = output.argmax(dim=1, keepdim=True)
            # Check if the adversarial sample fooled the classifier (e.g., expecting a different prediction)
            if pred.item() == 1:  # Adjust this check based on actual requirements
                correct += 1
    robustness_score = correct / total * 100  # Percentage of correct predictions
    print(f"Robustness Score: {robustness_score:.2f}%")
    return robustness_score
    
robustness_score = robustness_test(classifier_model, generator, z_dim)
