<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Shifting Narratives</h3>

  <p align="center">
    A repository containing the source code for the academic paper, "Shifting Narratives: A Dynamic Topic Modelling Analysis of Russia-Backed Tweets"
    <br />
    <a href="https://github.com/KieronHolmes/ShiftingNarratives"><strong>Read the paper »</strong></a>
  </p>
</div>

## About The Project

This repository houses the code and assets from a paper that presents findings from a dynamic topic modelling analysis of Tweets posted by Russian state-backed accounts operated by the Internet Research Agency (IRA) in St Petersburg. An unlabelled dataset released by Twitter spanning activity from 2012 to 2018 was used and we aimed to identify the different topics and their evolution over time. A BERT-based representation of the text was generated and dynamic topic modelling was applied for this purpose. Using HDBScan, a total of 141 topics were identified, each containing a cluster size of over 500. The top 20 of these topics were our main focus, accounting for 316,496 (16.53\%) of the total number of Tweets. An analysis was undertaken, finding that the majority of the Tweets focused on large sociopolitical events occurring between 2013 and 2018, and predominantly surrounded the creation of discourse and spread of misinformation in/about countries including the United States, United Kingdom, Ukraine, Syria and Iraq. Prominent themes identified and analysed include the Black Lives Matter Movement, 2016 Presidential Elections, Sports, Conflict in Iraq and Turkey, and British Politics.

## Getting Started

### Requirements
- Python 3.11
- CuDF
- CuML
- CUDA
- Pandas
- Numpy
- Matplotlib
- Nltk
- Scikit-Learn
- Click
- Sentence Transformers
- Wordcloud
- Pytorch
- OpenAI
- Tiktoken
- Bertopic (Spacy)
- Kaleido

### Cloning Repository
```
git clone https://github.com/KieronHolmes/ShiftingNarratives.git
cd ShiftingNarratives
```

### Configure Conda Environment

**Configure Environment (GPU):**
```
conda create -n shifting-narratives -c rapidsai -c conda-forge -c nvidia  \
    cudf=24.12 cuml=24.12 python=3.11 'cuda-version>=12.0,<=12.5' \
    pandas numpy matplotlib nltk scikit-learn click sentence-transformers wordcloud \
    'pytorch=*=*cuda*'
conda activate shifting-narratives
pip install openai tiktoken bertopic[spacy] kaleido
```

**Please Note:** The above setup command will allow you to configure the application in a GPU enabled fashion using Nvidia CUDA v12.0-12.5. If you are using differing versions, please see the [RapidsAI website](https://docs.rapids.ai/install/) for more info.

**Download the SpaCy PartOfSpeech tagger**
```
python -m spacy download en_core_web_sm
```

## Usage

Execute the `main.py` file in the repository, using the `input` flag for the CSV input for the application & `outputdir` to represent the directory where models, wordclouds and associated data should be saved.

**Please Note:** An OpenAI API Key must be supplied in order to generate OpenAI GPT topic representations.

**Example:**
```
python main.py --input=content.csv --outputdir=./output
```

## Input

Data used within this application should have (at minimum), the following columns: tweet_text, tweet_time, tweet_language & is_retweet.

Within this paper, we have utilised the 2018_IRA dataset produced by the [Twitter Moderation Research Consortium](https://web.archive.org/web/20240219201436/https://transparency.twitter.com/en/reports/moderation-research.html). Any of the `*_tweets.csv` datasets should work with this application; however, tuning of hyperparameters may be required for datasets with low volumes of Tweets and a likely low number of coherent topic clusters.

_**n.b.** The link provided above to the Twitter Moderation Research Consortium is a link via the Internet Archive service. Since the Twitter to X rebrand, moderation and transparency policies have changed, meaning this webpage is no longer publicly available._

## Output

- `./{output_directory}/csv` - Outputs `topic-info-all.csv` and `document-info-all.csv`
- `./{output_directory}/fivetopics` - If selected, will output Dynamic Topic models for each combination of topics for publication.
- `./{output_directory}/images` - Outputs `top_20_representation.svg` and `top_20_representation.html` (DTM representing top 20 topics by volume). Visualisations also output for each of the top 20 topics in formats `topic_{topicno}.svg` and `topic_{topicno}.html`
- `./{output_directory}/model` - Ouputs the model tensors
- `./{output_directory}/wordclouds` - Outputs wordclouds representing unigrams, bigrams and trigrams

## Notes

This application will automatically detect if `cuML` is installed in the Conda environment. If so, GPU acceleration will be used.
