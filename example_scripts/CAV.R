## Correlational Anchored Vector - Example Code

library(tidyverse)

# 1. Embed Texts of Interest

reddit_emotion <- read_csv("example_data/reddit_emotion.csv")

reddit_emotion_bge <- textEmbed(
  reddit_emotion$text,
  model = "BAAI/bge-base-en-v1.5", # model name
  tokens_select = "[CLS]", # select only [CLS] token embedding
  layers = -1,  # last layer
  dim_name = FALSE,
  keep_token_embeddings = FALSE
)

reddit_emotion_bge <- reddit_emotion_bge$texts[[1]]

write_csv(reddit_emotion_bge, "example_data/reddit_emotion_bge.csv")

# 2. Embed Training Data

source("example_data/load_power_narratives.R")

d_train <- read_csv("example_data/power_narratives.csv") |> 
  mutate(condition = factor(condition, levels = c("Control", "Power")))

d_train_bge <- textEmbed(
  d_train$text,
  model = "BAAI/bge-base-en-v1.5", # model name
  tokens_select = "[CLS]", # select only [CLS] token embedding
  layers = -1,  # last layer
  dim_name = FALSE,
  keep_token_embeddings = FALSE
)

d_train_bge <- d_train_bge$texts[[1]]

write_csv(d_train_bge, "example_data/d_train_bge.csv")

# 3. Train Correlational Anchored Vector

  # Rejoin embeddings to labelled training data
  d_train <- d_train |> 
    bind_cols(d_train_bge)
  
  # Partial Least Squares (PLS) regression
  library(caret)
  
  set.seed(2024)
  pls_control <- train(
    condition ~ ., 
    data = select(d_train, condition, Dim1:Dim768), 
    method = "pls",
    scale = FALSE,  # keep original embedding dimensions
    tuneLength = 1  # only 1 component (our anchored vector)
  )
  
  # Extract Projection 
  # (multiplied by loading to ensure positive correlation with the outcome)
  projection <- pls_control$finalModel$projection[,1]
  power_loading <- pls_control$finalModel$Yloadings["Power",1]
  power_cav <- projection*power_loading

# 4. Calculate Distance Metrics

reddit_emotion_control <- reddit_emotion_bge |> 
  rowwise() |> 
  mutate(control = dot_prod(c_across(Dim1:Dim768), power_cav)) |> 
  pull(control)

# 5. Test Hypothesis
  
t.test(
  x = reddit_emotion_control[reddit_emotion$subreddit=="depression"],
  y = reddit_emotion_control[reddit_emotion$subreddit=="TodayIamHappy"]
)
# r/TodayIamHappy has more power (as opposed to lack of power)

