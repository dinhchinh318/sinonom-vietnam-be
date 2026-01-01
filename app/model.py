import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class TokenClsService:
    def __init__(self, model_path: str, label_map_path: str, device: str = "auto"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"Label map not found: {label_map_path}")

        with open(label_map_path, "r", encoding="utf-8") as f:
            label2id = json.load(f)

        # label_map trong notebook của bạn thường là { "label": id }
        self.id2label = {int(v): k for k, v in label2id.items()}

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"input": "", "tokens": [], "labels": [], "output": ""}

        # Notebook của bạn dùng list(text) => tách theo ký tự
        tokens = list(text)

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc)
        pred_ids = torch.argmax(out.logits, dim=-1)[0].tolist()

        # Map label theo word_ids để tránh subword
        word_ids = enc["input_ids"].new(enc["input_ids"].shape).cpu()  # dummy holder
        # transformers cung cấp word_ids qua tokenizer(...) nhưng ở tensor dict không có sẵn,
        # nên cách ổn nhất: gọi tokenizer lần nữa dạng BatchEncoding để lấy word_ids.
        be = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        wi = be.word_ids(batch_index=0)

        labels = []
        aligned_tokens = []
        seen_word = set()

        for i, w in enumerate(wi):
            if w is None:
                continue
            if w in seen_word:
                continue
            seen_word.add(w)
            label = self.id2label.get(int(pred_ids[i]), "[UNK]")
            labels.append(label)
            aligned_tokens.append(tokens[w])

        output = " ".join(labels)  # nếu label chính là âm tiết/quốc ngữ

        return {
            "input": text,
            "tokens": aligned_tokens,
            "labels": labels,
            "output": output,
            "device": self.device,
        }
