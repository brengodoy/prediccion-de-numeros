# 🔢 Handwritten Digit Classifier

This is a beginner-friendly project where I built a simple digit recognition app using a neural network trained on the MNIST dataset. The goal was to understand the end-to-end workflow of a Machine Learning project — from model creation to deployment in a simple graphical interface.

> 🧠 This was my first Machine Learning project!  
> You’ll notice I included **lots of comments in the code** to help myself understand each step.  
> The purpose was educational — to learn deeply, not just make it work. 😊

---

## 💡 How it works

- You draw a number (0–9) in a canvas with your mouse.
- The image is preprocessed and passed to a trained PyTorch model.
- The model predicts which number it thinks you drew.
- The result is displayed instantly!

---

## 🧠 Model

The model is a simple feedforward neural network implemented with PyTorch:

- **Input layer**: 28×28 = 784 pixels (flattened)
- **Hidden layer**: 15 neurons, ReLU activation
- **Output layer**: 10 neurons (digits 0–9), with raw scores (logits)

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 15),
            nn.ReLU(),
            nn.Linear(15, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)
```

The model was trained on the **MNIST** dataset.

---

## ✨ GUI App

- Built with **Tkinter** (Python's standard GUI library).
- You can draw directly on a canvas.
- The app converts your drawing into a 28×28 grayscale image, normalizes it, and passes it to the model.

---

## 📂 Files

- `model.py`: The neural network definition.
- `training.py`: Script to train the model.
- `app.py`: The graphical user interface.
- `model.pth`: Saved PyTorch model (trained weights).

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

> `tkinter` comes pre-installed with Python (especially on Windows). No need to install it with pip.

---

## 🚀 How to Run

1. Train the model (or use the included `model.pth`):
   ```bash
   python training.py
   ```

2. Run the app:
   ```bash
   python app.py
   ```

---

## 📸 Preview

![image](https://github.com/user-attachments/assets/b5ff9fbf-a329-4a49-a570-4782f3ccf5c0)


---

## 📝 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 💬 Final note

Thank you for checking out my project!  
This was an exciting start in my Machine Learning journey — feel free to reach out if you have suggestions or feedback 😊💖
