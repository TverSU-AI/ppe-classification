from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
from datasets import load_dataset
import evaluate
from transformers import TrainingArguments, Trainer
import numpy as np


def check(image, expected):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 999 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx], predicted_class_idx)
    print("Expected", expected)


image = Image.open('content/mydataset/test/crash helmet/Screenshot 2024-03-05 230759.jpg')
image2 = Image.open('content/mydataset/test/cowboy boot/Screenshot 2024-03-05 230432.jpg')
model_name = 'WinKawaks/vit-small-patch16-224'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)
check(image, 'crash helmet')
check(image2, 'cowboy boot')


def map_dataset(arr):
    arr['pixel_values'] = feature_extractor(images=[x.convert("RGB") for x in arr['image']], return_tensors="pt")[
        'pixel_values']
    translate = {0: 514, 1: 518}
    arr['label2'] = [translate[x] for x in arr['label']]
    return arr


dataset = load_dataset("imagefolder", data_dir="content/mydataset", drop_labels=False)
dataset = dataset.map(map_dataset, batched=True)
dataset = dataset.remove_columns('label').rename_column('label2', 'label')
print(dataset['test'][0]['label'])
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=50)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()
model = model.to('cpu')
check(image, 'crash helmet')
check(image2, 'cowboy boot')
