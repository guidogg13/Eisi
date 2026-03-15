import re

def clean_dataset(text):
    # 1) Rimuove spazi multipli
    text = re.sub(r"[ \t]+", " ", text)

    # 2) Rimuove righe vuote multiple
    text = re.sub(r"\n\s*\n+", "\n", text)

    # 3) Uniforma USER: e ASSISTANT:
    text = re.sub(r"user\s*:", "USER:", text, flags=re.IGNORECASE)
    text = re.sub(r"assistant\s*:", "ASSISTANT:", text, flags=re.IGNORECASE)

    # 4) Rimuove caratteri strani non necessari
    text = re.sub(r"[^\wÀ-ÿ.,:;?!\n ]", "", text)

    # 5) Rimuove spazi all’inizio e alla fine delle righe
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)

    # 6) Separa i blocchi ASSISTANT → USER con una riga
    text = re.sub(r"(ASSISTANT:.*?)(?=USER:)", r"\1\n", text, flags=re.DOTALL)

    return text


if __name__ == "__main__":
    print("🔍 Leggo il dataset originale...")

    with open("data/dataset_raw.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("🧼 Pulizia in corso...")
    cleaned = clean_dataset(text)

    print("💾 Salvo il dataset pulito...")
    with open("data/dataset_clean.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)

    print("✅ Pulizia completata! File salvato in data/dataset_clean.txt")
