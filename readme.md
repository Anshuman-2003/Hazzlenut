# ♟️ ChessMovePredictor: Deep Learning Based Move Classifier

Welcome to **ChessMovePredictor**, a neural network-based chess move prediction engine that leverages FEN (Forsyth–Edwards Notation) to learn and predict the best move from large-scale human game data.

This project is the first phase of a larger goal: **building a complete chess engine combining deep learning with classical game tree search (Minimax)**.

---

## 🚀 Current Features

- ✅ **Parsed and Cleaned 4M+ moves** from a real-world chess dataset.
- ✅ Built a **move vocabulary** of 4.5k+ UCI notations.
- ✅ Implemented a **PyTorch model** trained on FEN + UCI move pairs.
- ✅ Achieved:
  - **Top-1 Accuracy**: ~21%
  - **Top-3 Accuracy**: ~39%
  - **Top-5 Accuracy**: ~48%
- ✅ Added extensive logs, error handling, and visualizations.
- ✅ Plotted **confusion matrix** on sampled predictions.
- ✅ Tested on a clean 200k sample subset for RAM efficiency.

---

🔍 Predict Moves From FEN

If you're using a virtual environment, you can install all requirements with:
pip install -r requirements.txt

To get the top-3 predicted UCI moves for any given board state:
Inside predict.py, you can replace the sample test_fen with any legal FEN position:
test_fen = "r1bqkbnr/pppppppp/n7/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
in termnal hit "python model/predict_moves.py"
This will output something like:
Top predicted moves:
g8f6: 0.3624
b8c6: 0.2141
d7d6: 0.1333

___

### Input
- FEN string → converted into a **(13×8×8)** tensor:
  - 12 channels for piece planes
  - 1 for side to move

### Model
```python
ChessMoveClassifier(
  Conv2D → ReLU → MaxPool
  Conv2D → ReLU → MaxPool
  Flatten → Linear → ReLU → Dropout
  Linear → Output (softmax over ~4544 moves)
)

---

Project Structure(Just for the AI model)

client/ #Will have UI for chess engine
data/ #All data raw,processed and final
model/
├── dataset.py          # Custom PyTorch Dataset (FEN + UCI)
├── model.py            # CNN-based classifier
├── train.py            # Full training loop with logging
├── test.py             # Evaluation, accuracy, confusion matrix
├── utils.py            # FEN encoding, vocab creation
├── move_vocab.json     # UCI → idx and idx → UCI
├── train.csv           # Raw training data
├── train_cleaned.csv   # Cleaned & filtered training data
├── test.csv            # Raw test data
├── test_cleaned.csv    # Cleaned & filtered test data
└── chess_model.pt      # 🔥 Trained model
temp/ #Temprory folder
utils/ #Contains functions for data preprocessing 

🧪 Next Goals
🔄 Integrate the model with a Minimax engine to improve search.
♟️ Build an interactive GUI to play against the AI.
📈 Hyperparameter tuning & deeper CNN variants.
📦 Add support for different difficulty levels.

🛠️ Tech Stack
Language: Python 3.11
Framework: PyTorch
Tools: Kaggle Notebooks, Pandas, Seaborn, Matplotlib
GPU: CUDA (T4, P100 for training)
Data Format: FEN + UCI

🧠 Inspiration
This project draws inspiration from AlphaZero, but the approach is intentionally educational: combining traditional AI (Minimax) with learned policies (move prediction) to create a hybrid, explainable chess engine.

🤝 Contribution
Want to help improve it? Have ideas? Open a PR or reach out!

💬 Author
Anshuman (SpunkY) – Engineer & Chess AI Enthusiast 👨‍💻
Find me on GitHub, Kaggle, or LinkedIn!

📌 Note
This model is trained to predict the most likely move played by humans, not necessarily the best move. It forms a policy network for guiding smarter searche

© License
MIT License – free to use and modify. Please give credit if you build upon it!

