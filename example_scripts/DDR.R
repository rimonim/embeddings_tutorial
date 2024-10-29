## Distributed Dictionary Representation (DDR) - Example Code

library(tidyverse)
library(quanteda)
source("embedding_scripts.R")

# 1. Load Word Embedding Model

path_to_glove <- "~/Projects/ds4psych/data/glove/glove.twitter.27B.100d.txt"

glove_dimensions <- as.numeric(str_extract(path_to_glove, "[:digit:]+(?=d\\.txt)"))

glove_pretrained <- data.table::fread(
  path_to_glove, 
  quote = "",
  col.names = c("token", paste0("dim_", 1:glove_dimensions))
  ) |> 
  distinct(token, .keep_all = TRUE) |> 
  remove_rownames() |> 
  column_to_rownames("token") |> 
  as.matrix()

class(glove_pretrained) <- "embeddings"

# 2. Embed Texts of Interest

reddit_emotion <- read_csv("example_data/reddit_emotion.csv")

reddit_emotion_corp <- corpus(
  reddit_emotion, 
  docid_field = "doc_id", 
  text_field = "text"
  )

reddit_emotion_dfm <- reddit_emotion_corp |> 
  tokens(remove_punct = TRUE, remove_url = TRUE) |> 
  dfm()

reddit_emotion_glove <- reddit_emotion_dfm |> 
  textstat_embedding(glove_pretrained) |> 
  select(-doc_id)

# 3. Load and Embed Dictionaries

  # Format and export tokens to run through LIWC software
  reddit_emotion_tokens <- featnames(reddit_emotion_dfm)
  write_csv(tibble(text = reddit_emotion_tokens), "~/Downloads/reddit_emotion_tokens.csv")

  # Load expanded LIWC dictionaries
  reddit_emotion_tokens <- read_csv("~/Downloads/reddit_emotion_tokens_LIWC.csv") |> drop_na()
  
  i_dict <- reddit_emotion_tokens$text[reddit_emotion_tokens$i > 0]
  shehe_they_dict <- reddit_emotion_tokens$text[reddit_emotion_tokens$shehe > 0 | reddit_emotion_tokens$they > 0]

  # Remove tokens that are not in embedding model
  i_dict <- i_dict[i_dict %in% rownames(glove_pretrained)]
  shehe_they_dict <- shehe_they_dict[shehe_they_dict %in% rownames(glove_pretrained)]
  
  # Embed dictionary words
  i_dict_glove <- i_dict |> 
    tokens() |> 
    dfm() |> 
    textstat_embedding(glove_pretrained) |> 
    select(-doc_id) |> 
    as.matrix()
  rownames(i_dict_glove) <- i_dict
  
  shehe_they_dict_glove <- shehe_they_dict |> 
    tokens() |> 
    dfm() |> 
    textstat_embedding(glove_pretrained) |> 
    select(-doc_id) |> 
    as.matrix()
  rownames(shehe_they_dict_glove) <- shehe_they_dict
  
# 4. Get Dictionary Word Frequencies

i_dict_freqs <- reddit_emotion_dfm |> 
  dfm_keep(i_dict) |> 
  quanteda.textstats::textstat_frequency() |> 
  pull(frequency, name = feature)

shehe_they_dict_freqs <- reddit_emotion_dfm |> 
  dfm_keep(shehe_they_dict) |> 
  quanteda.textstats::textstat_frequency() |> 
  pull(frequency, name = feature)
  
# 5. Compute Anchored Vector

  # Aggregate dictionary embeddings with weighted average
  i_DDR <- apply(i_dict_glove, 2, weighted.mean, w = i_dict_freqs)
  shehe_they_DDR <- apply(shehe_they_dict_glove, 2, weighted.mean, w = shehe_they_dict_freqs)

  # Anchored vector
  selfref_anchored <- i_DDR - shehe_they_DDR

# 6. Calculate Distance Metrics

  reddit_emotion_scores <- reddit_emotion_glove |> 
    rowwise() |> 
    mutate(selfref = dot_prod(c_across(V1:V100), selfref_anchored)) |> 
    pull(selfref)

# 7. Test Hypothesis (two sample t-test)

t.test(
  x = reddit_emotion_scores[reddit_emotion$subreddit=="depression"],
  y = reddit_emotion_scores[reddit_emotion$subreddit=="TodayIamHappy"]
  )
  # r/depression has more self-referential (as opposed to other-referential) language
