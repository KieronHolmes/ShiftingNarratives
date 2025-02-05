# Argument Parsing
import argparse
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="The input csv location")
parser.add_argument("--outputdir", help="The output directory")
args = parser.parse_args()

# Add the NLTK_Data Item to Path
nltk.data.path.append("./nltk_data/")
nltk.download("punkt_tab", download_dir="./nltk_data")

# Validate the parameters
if args.input is None or args.outputdir is None:
    print("Input or OutputDir parameters are not present")
    sys.exit(0)
elif not (os.path.isfile(args.input) or args.input.endswith("csv")):
    print("Input file must be a csv")
    sys.exit(0)

# Make directory if does not exist
if not os.path.isdir(args.outputdir):
    print("Creating output directory")
    os.mkdir(args.outputdir)
    Path("./output/csv/").mkdir(parents=True, exist_ok=True)
    Path("./output/images/").mkdir(parents=True, exist_ok=True)
    Path("./output/wordclouds/").mkdir(parents=True, exist_ok=True)
    Path("./output/model/").mkdir(parents=True, exist_ok=True)
    Path("./output/fivetopics/").mkdir(parents=True, exist_ok=True)

#############################
# GPU Optimise if Available #
#############################
try:
    import cuml
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP

    use_gpu = True
    print("Using GPU for UMAP and HDBSCAN")
except ImportError:
    from hdbscan import HDBSCAN
    from umap import UMAP

    use_gpu = False
    print("Using CPU for UMAP and HDBSCAN")


######################
# Reusable Functions #
######################
def clean_tweets(tweet_text):
    tweet_text = re.sub(
        r"http\S+|www\S+|https\S+", "", str(tweet_text), flags=re.IGNORECASE
    )  # Remove URLs
    tweet_text = re.sub(r"@\w+", "", tweet_text)  # Remove user mentions
    tweet_text = re.sub(r"#", "", tweet_text)  # Remove Hashtag Symbol

    tokens = nltk.word_tokenize(tweet_text)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words("english")]

    return " ".join(tokens)


def generate_word_cloud(data, filename):
    wordcloud = WordCloud(
        width=3000, height=2000, background_color="black"
    ).generate_from_frequencies(data)
    fig = plt.figure(figsize=(40, 30), facecolor="k", edgecolor="k")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(
        "{directory}/wordclouds/{fname}.png".format(
            directory=args.outputdir, fname=filename
        )
    )


def generate_ngrams(tokens, n):
    n_grams = ngrams(tokens, n)
    return [" ".join(gram) for gram in n_grams]


##################
# Core code here #
##################

# Pandas datatype for file ingression
dtypes_tweet = {
    "tweet_text": str,
    "tweet_time": str,
    "tweet_language": str,
    "is_retweet": bool,
}

# Load dataframes and set fields to be imported as dates
df = pd.read_csv(
    args.input,
    usecols=["tweet_text", "tweet_time", "tweet_language", "is_retweet"],
    dtype=dtypes_tweet,
    low_memory=True,
)
print("Dataframe `df` has been initialised")

# Output basic stats
print("{} Records loaded into Dataframe `df`".format(df.shape[0]))

print("Before:")
print(df.info())

# Limit for dev
# df = df[:50000]

#################
# Preprocessing #
#################

# Drop non-English tweets
df.drop(df[df.tweet_language != "en"].index, inplace=True)

# Drop Retweets
df.drop(df[df.is_retweet == True].index, inplace=True)

# Apply Preprocessing
df["tweet_text"] = df["tweet_text"].apply(lambda x: clean_tweets(x))

# Remove Empty Tweets
df["tweet_text"].replace("", np.nan, inplace=True)
df.dropna(subset=["tweet_text"], inplace=True)

# Create Static Lists
tweets = df["tweet_text"].to_list()
timestamps = df["tweet_time"].to_list()
print("Created static list named `tweets` and `timestamps`")

print("After:")
print(df.info())

# Tokens
tokens = df["tweet_text"].apply(word_tokenize)
print("Tokenised and saved as `tokens` list")

# Unigrams
ngram_list = [gram for token in tokens for gram in generate_ngrams(token, 1)]

generate_word_cloud(nltk.FreqDist(ngram_list), "unigram_wordcloud")
print("Generated unigrams render")

# Bigrams
ngram_list = [gram for token in tokens for gram in generate_ngrams(token, 2)]

generate_word_cloud(nltk.FreqDist(ngram_list), "bigram_wordcloud")
print("Generated bigrams render")

# Trigrams
ngram_list = [gram for token in tokens for gram in generate_ngrams(token, 3)]

generate_word_cloud(nltk.FreqDist(ngram_list), "trigram_wordcloud")
print("Generated trigrams render")

# Generate Embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(tweets, show_progress_bar=True, device=device)
print("Generated embeddings")

# Vectorizer Model
vectorizer_model = CountVectorizer()
print("Created vectorizer model")

# CF-TF-IDF Model
ctfidf_model = ClassTfidfTransformer()
print("Created CF TF-IDF Model")

# UMAP
if use_gpu:
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
else:
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
print("Created UMAP Model")


# HDBSCAN
if use_gpu:
    hdbscan_model = HDBSCAN(
        min_samples=10,
        gen_min_span_tree=True,
        min_cluster_size=500,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
        verbose=True,
    )
else:
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
print("Created HDBSCAN Model")

##########
# OpenAI #
##########
import openai
import tiktoken
from bertopic.representation import OpenAI

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
client = openai.OpenAI(api_key="insert-openai-api-key-here")
OpenAIModel = OpenAI(
    client,
    model="gpt-4o-mini",
    delay_in_seconds=0.5,
    exponential_backoff=True,
    chat=True,
    nr_docs=15,
    doc_length=280,
    tokenizer=tokenizer,
)

########################
# BERTopic Topic Model #
########################
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    PartOfSpeech,
)

pos_model = PartOfSpeech("en_core_web_sm")
mmr_model = MaximalMarginalRelevance(diversity=0.3)
keybert_model = KeyBERTInspired()


########################
# Representation Model #
########################
representation_model = {
    "KeyBERT": keybert_model,
    "MMR": mmr_model,
    "POS": pos_model,
    "OpenAI": OpenAIModel,
}

########################
# BERTopic Topic Model #
########################

# Create Topic Model
topic_model = BERTopic(
    verbose=True,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,
    low_memory=True,
    embedding_model=sentence_model,
)
print("Created and trained BERTopic Model")

# Get Topics and Probabilities
topics, probs = topic_model.fit_transform(documents=tweets, embeddings=embeddings)
print("Fit transform complete")

# Output Topic Info
topic_model.get_topic_info()

# Save Topic Model
topic_model.save(
    "{directory}/model/".format(directory=args.outputdir),
    serialization="safetensors",
    save_ctfidf=True,
    save_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)
print("Saved Model")

print("====================\nMAIN RUN COMPLETED\n====================")

####################
# Save Output Data #
####################
topic_info = topic_model.get_topic_info()
topic_info.to_csv("./output/csv/topic-info-all.csv", index=False)

document_info = topic_model.get_document_info(tweets)
document_info.to_csv("./output/csv/document-info-all.csv", index=False)

##################
# Topic Labels   #
##################

topic_labels = topic_model.generate_topic_labels(aspect="OpenAI")
topic_model.set_topic_labels(topic_labels)

##################
# Visualisations #
##################
topicsOverTime = topic_model.topics_over_time(tweets, timestamps, nr_bins=20)

img_out = topic_model.visualize_topics_over_time(
    topicsOverTime, top_n_topics=20, custom_labels=True
)
img_out.write_image("./output/images/top_20_representation.svg")
img_out.write_html("./output/images/top_20_representation.html")

for iter in range(0, 20):
    img_out = topic_model.visualize_topics_over_time(
        topicsOverTime, topics=[iter], custom_labels=True
    )
    img_out.write_image("./output/images/topic_{topicno}.svg".format(topicno=iter))
    img_out.write_html("./output/images/topic_{topicno}.html".format(topicno=iter))

import click

if click.confirm("Do you want to generate combinational outputs?", default=False):
    # Generate Combinations of 5
    import itertools

    items = list(range(20))
    combinations = itertools.combinations(items, 5)

    for combination in combinations:
        folder_dir = combination_str = "_".join(map(str, combination))
        Path("./output/images/fivetopics/{folder}/".format(folder=folder_dir)).mkdir(
            parents=True, exist_ok=True
        )
        img_out = topic_model.visualize_topics_over_time(
            topicsOverTime, topics=list(combination), custom_labels=True
        )
        img_out.write_image(
            "./output/images/fivetopics/{folder}/topic_graph.svg".format(
                folder=folder_dir, topicno=iter
            )
        )
        img_out.write_html(
            "./output/images/fivetopics/{folder}/topic_graph.html".format(
                folder=folder_dir, topicno=iter
            )
        )
