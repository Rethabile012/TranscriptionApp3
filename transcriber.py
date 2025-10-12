import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration, get_scheduler
from torch.optim import AdamW
from evaluate import load
import pandas as pd
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# ------------------ CONFIG ------------------
TRAIN_DIR = "TrainingSet"
VAL_DIR = "ValidationSet"
TEST_DIR = "TestSet"
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 1e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ DATASET ------------------
class AudioTranscriptionDataset(Dataset):
    def __init__(self, audio_dir, transcript_dir, processor):
        self.samples = []
        self.processor = processor

        for date_folder in os.listdir(audio_dir):
            date_audio_path = os.path.join(audio_dir, date_folder)
            date_text_path = os.path.join(transcript_dir, date_folder)
            if not os.path.isdir(date_audio_path):
                continue

            for file in os.listdir(date_audio_path):
                if file.endswith(".wav"):
                    audio_path = os.path.join(date_audio_path, file)
                    txt_path = os.path.join(date_text_path, file.replace(".wav", ".txt"))
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            transcript = f.read().strip()
                        self.samples.append((audio_path, transcript))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, transcript = self.samples[idx]
        speech_array, sampling_rate = torchaudio.load(audio_path)
        speech_array = torchaudio.functional.resample(speech_array, sampling_rate, 16000).squeeze()

        # Process the audio
        input_features = self.processor.feature_extractor(
            speech_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features[0]

        # Process the text (tokenize labels)
        labels = self.processor.tokenizer(
            transcript,
            return_tensors="pt",
            padding="longest",
            truncation=True
        ).input_ids[0]

        return input_features, labels



# ------------------ COLLATE FUNCTION ------------------
def collate_fn(batch):
    input_features, labels = zip(*batch)

    input_features_padded = pad_sequence(input_features, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 ignored by loss

    return input_features_padded, labels_padded


# ------------------ SETUP ------------------
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
cer_metric = load("cer")

train_dataset = AudioTranscriptionDataset(
    os.path.join(TRAIN_DIR, "audio"),
    os.path.join(TRAIN_DIR, "transcripts"),
    processor,
)
val_dataset = AudioTranscriptionDataset(
    os.path.join(VAL_DIR, "audio"),
    os.path.join(VAL_DIR, "transcripts"),
    processor,
)
test_dataset = AudioTranscriptionDataset(
    os.path.join(TEST_DIR, "audio"),
    os.path.join(TEST_DIR, "transcripts"),
    processor,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, num_workers=1)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

train_losses, val_losses, val_cers = [], [], []
best_cer = float("inf")
best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")


# ------------------ TRAINING LOOP ------------------
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        input_features, labels = batch
        input_features = input_features.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Training Loss: {avg_train_loss:.4f}")

    # ------------------ VALIDATION ------------------
    model.eval()
    val_loss = 0.0
    cer_total = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_features, labels = batch
            input_features = input_features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_features=input_features, labels=labels)
            val_loss += outputs.loss.item()

            pred_ids = model.generate(input_features)
            pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.batch_decode(labels, skip_special_tokens=True)
            cer_total += cer_metric.compute(predictions=pred_str, references=label_str)

    avg_val_loss = val_loss / len(val_loader)
    avg_cer = cer_total / len(val_loader)
    val_losses.append(avg_val_loss)
    val_cers.append(avg_cer)

    print(f"Validation Loss: {avg_val_loss:.4f} | CER: {avg_cer:.4f}")

    if avg_cer < best_cer:
        best_cer = avg_cer
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved (CER: {best_cer:.4f})")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pt"))

# ------------------ SAVE TRAINING METRICS ------------------
df = pd.DataFrame({
    "Epoch": range(1, EPOCHS + 1),
    "TrainLoss": train_losses,
    "ValLoss": val_losses,
    "ValCER": val_cers
})
df.to_csv("loss_history.csv", index=False)
print("\nTraining complete!")
print(f"Best model saved at: {best_model_path} (CER: {best_cer:.4f})")

# ------------------ TEST EVALUATION ------------------
print("\nEvaluating best model on TEST SET...")
best_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
best_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
best_model.eval()

test_cer_total = 0.0
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_features, labels = batch
        input_features = input_features.to(DEVICE)
        labels = labels.to(DEVICE)

        pred_ids = best_model.generate(input_features)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(labels, skip_special_tokens=True)
        test_cer_total += cer_metric.compute(predictions=pred_str, references=label_str)

avg_test_cer = test_cer_total / len(test_loader)
print(f"\n🧪 Test Set CER: {avg_test_cer:.4f}")
