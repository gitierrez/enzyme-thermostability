import os
import evaluate

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.integrations import TensorBoardCallback

from src.data import load_dataset


model_checkpoint = 'facebook/esm2_t6_8M_UR50D'


# check if processed_data directory exists and contains data
if os.path.exists('processed_data'):
    ds = load_from_disk('processed_data')
else:
    df = load_dataset('data/')[['protein_sequence', 'tm']]
    # only keep sequences shorter than 1k
    df = df[df['protein_sequence'].str.len() < 1000]
    ds = Dataset.from_pandas(df)
    ds = ds.rename_columns({'protein_sequence': 'text', 'tm': 'label'})

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


    def tokenize_function(examples):
        # TODO: dynamic padding
        return tokenizer(examples['text'], padding='max_length', truncation=True)


    ds = ds.map(tokenize_function, batched=True)
    ds = ds.train_test_split(test_size=0.1)
    ds.save_to_disk('processed_data')

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1).to('cuda')

# freeze body
for param in model.esm.parameters():
    param.requires_grad = False

effective_batch_size = 16
oom_batch_size = 4
training_args = TrainingArguments(
    output_dir="seq-only",
    evaluation_strategy="epoch",
    num_train_epochs=5,
    gradient_accumulation_steps=effective_batch_size // oom_batch_size,
    per_device_train_batch_size=oom_batch_size,
    per_device_eval_batch_size=oom_batch_size,
)

metric = evaluate.load('mae')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.squeeze(-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    compute_metrics=compute_metrics,
    callbacks=[TensorBoardCallback()]
)

trainer.train()

print("Done!")
