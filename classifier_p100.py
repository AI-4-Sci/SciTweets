import pandas as pd
import ast
from functools import partial
import numpy as np
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score
from emoji import demojize
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()


def normalize_tweet(tweet):
    normTweet = (
        tweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return normTweet


def replace_user_handles(tweet, replace='@USER'):
    tokens = tokenizer.tokenize(tweet)

    new_tokens = []
    for token in tokens:
        if token.startswith("@"):
            new_tokens.append(replace)
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)


def replace_urls(tweet, replace='HTTPURL'):
    tokens = tokenizer.tokenize(tweet)

    if type(replace) == str:
        new_tokens = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token.startswith("http") or lower_token.startswith("www"):
                new_tokens.append(replace)
            else:
                new_tokens.append(token)

    elif type(replace) == list:
        n_replaced_tokens = 0
        new_tokens = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token.startswith("http") or lower_token.startswith("www"):
                if n_replaced_tokens < len(replace):
                    new_tokens.append(replace[n_replaced_tokens])
                    n_replaced_tokens = n_replaced_tokens + 1
                else:
                    new_tokens.append('')
            else:
                new_tokens.append(token)

    return " ".join(new_tokens)


def replace_emojis(tweet, replace='demojize'):
    tokens = tokenizer.tokenize(tweet)

    new_tokens = []
    for token in tokens:
        if len(token) == 1:
            if replace == 'demojize':
                new_tokens.append(demojize(token))
            else:
                new_tokens.append(replace)
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def compute_metrics(x):
    print('compute metrics')
    predictions = (x.predictions > 0) * 1
    labels = x.label_ids
    cat1_acc = accuracy_score(labels[:,0], predictions[:,0])
    cat2_acc = accuracy_score(labels[:,1], predictions[:,1])
    cat3_acc = accuracy_score(labels[:,2], predictions[:,2])

    cat1_pre = precision_score(labels[:, 0], predictions[:, 0])
    cat2_pre = precision_score(labels[:, 1], predictions[:, 1])
    cat3_pre = precision_score(labels[:, 2], predictions[:, 2])

    cat1_re = recall_score(labels[:, 0], predictions[:, 0])
    cat2_re = recall_score(labels[:, 1], predictions[:, 1])
    cat3_re = recall_score(labels[:, 2], predictions[:, 2])

    return {'cat1_acc': cat1_acc, 'cat2_acc': cat2_acc, 'cat3_acc': cat3_acc,
            'cat1_pre': cat1_pre, 'cat2_pre': cat2_pre, 'cat3_pre': cat3_pre,
            'cat1_re': cat1_re, 'cat2_re': cat2_re, 'cat3_re': cat3_re}


if __name__ == '__main__':
    model_id = 'scibert_uncased'

    tokenizer_config = {'pretrained_model_name_or_path': 'allenai/scibert_scivocab_uncased',
                      'max_len': 420}

    model_config = {'pretrained_model_name_or_path': 'allenai/scibert_scivocab_uncased',
                    'num_labels': 3,
                    'problem_type': 'multi_label_classification'}

    dataloader_config = {'per_device_train_batch_size': 8,
                         'per_device_eval_batch_size': 8}

    preprocessing_config = {'lowercase': True,
                             'normalize': True,
                             'urls': 'original_urls',
                             'user_handles': '@USER',
                             'emojis': 'demojize'}

    predicted_eval_data = []

    data = pd.read_csv('annotations/annotations.tsv', sep='\t')
    data['processed_urls'] = data['processed_urls'].apply(lambda urls: ast.literal_eval(urls))
    data['label'] = data[['cat1_final_answer', 'cat2_final_answer', 'cat3_final_answer']].apply(
        lambda x: [x[0], x[1], x[2]], axis=1)
    data['original_text'] = data['text']

    data = data[~data['label'].astype(str).str.contains("0.5")]

    train_data = data[data['round'] == 1]
    eval_data = data[data['round'] == 2]

    train_data_tids = set(train_data['tweet_id'].tolist())
    eval_data_tids = set(eval_data['tweet_id'].tolist())

    assert len(train_data_tids.intersection(eval_data_tids)) == 0

    print('load model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
    model = AutoModelForSequenceClassification.from_pretrained(**model_config).cuda()

    print('preprocess tweets')
    def preprocess_function(data):
        if preprocessing_config['lowercase']:
            data["text"] = data["text"].str.lower()

        if preprocessing_config['normalize']:
            data["text"] = data["text"].apply(normalize_tweet)

        if preprocessing_config['emojis'] != False:
            data["text"] = data["text"].apply(partial(replace_emojis, replace=preprocessing_config['emojis']))

        if preprocessing_config['user_handles'] != False:
            data["text"] = data["text"].apply(
                partial(replace_user_handles, replace=preprocessing_config['user_handles']))

        if preprocessing_config['urls'] == 'original_urls':
            data["text"] = data[["text", 'processed_urls']].apply(lambda row: replace_urls(*row), axis=1)

        elif preprocessing_config['urls'] != False:
            data["text"] = data["text"].apply(partial(replace_urls, replace=preprocessing_config['urls']))

        return data

    train_data = preprocess_function(train_data)
    eval_data = preprocess_function(eval_data)

    print('transform data')
    train_dataset = Dataset.from_pandas(train_data[['tweet_id', 'text', 'label']])
    eval_dataset = Dataset.from_pandas(eval_data[['tweet_id', 'text', 'label']])

    print('tokenize tweets')
    def tokenize_function(examples):
        model_inputs = tokenizer(examples["text"], max_length=tokenizer_config['max_len'], truncation=True, padding='max_length')
        model_inputs["labels"] = examples['label']
        return model_inputs

    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=None)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, batch_size=None)

    print('set up Trainer')
    training_args = TrainingArguments(
        output_dir="classifier_preds",  # output directory
        num_train_epochs=3,  # total number of training epochs
        **dataloader_config,
        warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        save_strategy='no',
        evaluation_strategy="epoch",  # evaluate each `logging_steps`
        no_cuda=False,
        report_to='none'
    )

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()

    print('start inference')
    pred_output = trainer.predict(eval_dataset)
    predictions = (pred_output.predictions > 0) * 1

    eval_data['cat1_pred'] = predictions[:, 0]
    eval_data['cat2_pred'] = predictions[:, 1]
    eval_data['cat3_pred'] = predictions[:, 2]

    eval_data['cat1_score'] = sigmoid(pred_output.predictions[:, 0])
    eval_data['cat2_score'] = sigmoid(pred_output.predictions[:, 1])
    eval_data['cat3_score'] = sigmoid(pred_output.predictions[:, 2])

    eval_data.to_csv('classifier_preds/scibert_2stage_predictions_new.tsv', index=False, sep='\t')