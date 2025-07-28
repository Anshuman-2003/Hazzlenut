# ♟️ ChessMovePredictor: Deep Learning-Based Move Classifier

Welcome to **ChessMovePredictor**, a neural network-based chess move prediction engine that leverages **FEN (Forsyth–Edwards Notation)** to learn and predict the best move from large-scale human game data.

> This project is the first phase of a larger goal:  
> **Building a complete chess engine combining deep learning with classical game tree search (Minimax).**

---

## 🚀 Features

- ✅ **Parsed and cleaned 4M+ moves** from a real-world chess dataset  
- ✅ Built a **move vocabulary** of 4.5k+ UCI notations  
- ✅ Implemented a **PyTorch model** trained on FEN + UCI move pairs  
- ✅ Evaluation on ~1000 random positions:
  - **Average Eval Drop:** 2.38 centipawns  
  - **Moves within 50 cp drop:** 76.83%  
  - **Moves within 100 cp drop:** 82.32%
- ✅ Extensive logs, error handling, and visualizations  
- ✅ Tested on 1000 random positions with generated evaluation logs and histogram  

---

## 🔍 Generate Your Own Evaluation

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Install the Stockfish engine on your local system to test.

3. Prepare your evaluation data:
   - Generate a `FEN_move.csv` file containing 1000–2000 random FEN positions.

4. Edit the file path in `eval.py`:

   ```python
   CSV_PATH = "data/processed/oldDS/random_fens_1000.csv"  # ➜ Replace with your own CSV path
   ```

5. Update the Stockfish path:

   - Run `which stockfish` in your terminal.
   - Update:

     ```python
     STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # ➜ Replace with your local Stockfish path
     ```

6. Run the evaluation script:

   ```bash
   python model/eval.py
   ```

   This will output evaluation logs and generate a histogram.

---

🎮 Play Against the Chess AI

You can play a full game of chess against your AI-powered model However it only has positional understanding and no knowledge of
captures and piece value

▶️ Steps to Run

1.	Make sure dependencies are installed:
     ```bash
   pip install -r requirements.txt 
   ```
2.	Ensure the model and move vocabulary exist:
	•	model/chess_model_new.pt (your trained model)
	•	model/move_vocab.json (your move index mapping)
	
3.	Run the interactive game script:
     ```bash
   python model/play.py
   ```
4.	The board will be printed in the terminal after each move.
	•	You play as White, the AI plays as Black.
	•	Enter your move in UCI format (e.g. e2e4, g1f3, etc.).
	•	The AI combines a learned policy with a minimax search enhanced by capture pruning and quiescence evaluation.

⸻

🧠 How It Works
	•	The engine uses your trained model to rank top policy moves from FEN.
	•	For tactical sharpness, it prioritizes captures when possible.
	•	A depth-limited minimax with alpha-beta pruning and quiescence search chooses the best move for the bot.
	•	Evaluations are based on material score and position stability.

---

## 🧾 Model Input

- **Input**: FEN string → Converted into a **(13 × 8 × 8)** tensor:
  - 12 channels for piece planes
  - 1 channel for side to move

---

## 🧠 Model Architecture

```python
ChessMoveClassifier(
  Conv2D → ReLU → MaxPool
  Conv2D → ReLU → MaxPool
  Flatten → Linear → ReLU → Dropout
  Linear → Output (softmax over ~4544 moves)
)
```

---

## 📁 Project Structure (AI Model Only)

```
client/               # UI for the chess engine (Work in progress)
data/                 # All data - raw, processed, and final
engine/               # Minimax + model integration (WIP)
model/
├── dataset.py        # Custom PyTorch Dataset (FEN + UCI)
├── model.py          # CNN-based classifier
├── train.py          # Full training loop with logging
├── test.py           # Evaluation: accuracy, confusion matrix
├── utils.py          # FEN encoding, vocab creation
├── move_vocab.json   # UCI → idx and idx → UCI
└── chess_model_new.pt  # 🔥 Trained model
temp/                 # Temporary folder with locally trained models
utils/                # Data preprocessing functions
evaluation/           # Log file + histogram from Stockfish evals
```

---

## 🧪 Next Goals

- 🔄 Integrate the model with a Minimax engine  
- ♟️ Build an interactive GUI to play against the AI  
- 📈 Hyperparameter tuning and deeper CNN variants  
- 📦 Add support for different difficulty levels  

---

## 🛠️ Tech Stack

- **Language**: Python 3.11  
- **Framework**: PyTorch  
- **Tools**: Kaggle Notebooks, Pandas, Seaborn, Matplotlib  
- **GPU**: CUDA (T4, P100 for training)  
- **Data Format**: FEN + UCI  

---

## 🧠 Inspiration

This project draws inspiration from **AlphaZero**, but with a focus on **educational and hybrid explainable AI**:

> Combining traditional AI (Minimax) with learned policies (move prediction) to create a hybrid, interpretable chess engine.

---

## 🤝 Contribution

Have ideas? Found a bug? Want to help improve it?  
Open a PR or reach out!

---

## 👨‍💻 Author

**Anshuman (SpunkY)** – Engineer & Chess AI Enthusiast  
Find me on GitHub, Kaggle, or LinkedIn!

---

## 📌 Note

This model is trained to **predict the most likely move played by strong human players** (2500+ ELO), not necessarily the best move.  
It acts as a **policy network to guide smarter search**.

---

## 📜 License

MIT License – free to use and modify.  
Please give credit if you build upon it!
