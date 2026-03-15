import re
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def get_stats(self, words):
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_pair(self, pair, words):
        bigram = " ".join(pair)
        replacement = "".join(pair)
        new_words = {}
        for word, freq in words.items():
            new_word = re.sub(bigram, replacement, word)
            new_words[new_word] = freq
        return new_words

    def train(self, text):
        # Step 1: split words into characters
        words = text.split()
        words = [" ".join(list(w)) + " </w>" for w in words]

        # Count frequencies
        word_freqs = Counter(words)

        # BPE loop
        for i in range(self.vocab_size):
            pairs = self.get_stats(word_freqs)
            if not pairs:
                print("Nessuna coppia trovata, fine.")
                break

            best_pair, freq = pairs.most_common(1)[0]

            # 🔥 Stampa il merge corrente
            print(f"[{i}] Merge: {best_pair}  |  Frequenza: {freq}")

            # Salva il merge
            self.merges.append(best_pair)

            # Applica il merge
            word_freqs = self.merge_pair(best_pair, word_freqs)


        # Build vocab
        tokens = set()
        for word in word_freqs:
            for token in word.split():
                tokens.add(token)

        self.vocab = {token: i for i, token in enumerate(sorted(tokens))}

    def save(self, prefix="my_tokenizer"):
        with open(f"{prefix}_vocab.json", "w", encoding="utf-8") as f:
            f.write(str(self.vocab))

        with open(f"{prefix}_merges.txt", "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")
if __name__ == "__main__":
    print("Carico dataset...")

    with open("data/dataset_clean.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("Lunghezza dataset:", len(text))
    print("Avvio training BPE...")

    tokenizer = BPETokenizer(vocab_size=8000)
    tokenizer.train(text)

    print("Salvo i file...")
    tokenizer.save("eisi_bpe")

    print("Fatto!")
