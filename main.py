from Shakespeare import load_shakespeare
from TextDataset import TextDataset

if __name__ == "__main__":
    t = TextDataset("SNShakespeare")
    t.load_data()
    t.generate_vocabulary()