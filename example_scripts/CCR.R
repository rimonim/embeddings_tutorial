## Contextualized Construct Representation (CCR) - Example Code

library(tidyverse)
library(text)

# 1. Embed Texts of Interest
# @xiao_etal_2023

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

# 2. Embed Questionnaires

  # Questionnaire: Rotter (1966)
  # @rotter_1966
    # Positive (internal locus of control)

internal_items = c(
  "People's misfortunes result from the mistakes they make",
  "One of the major reasons why we have wars is because people don't take enough interest in politics",
  "In the long run people get the respect they deserve in this world",
  "The idea that teachers are unfair to students is nonsense",
  "Capable people who fail to become leaders hive not taken advantage of their opportunities",
  "People who can't get others to like them don't understand how to get along with others",
  "Trusting to fate has never turned out as well for me as making a decision to take a definite course of action",
  "In the case of the well prepared student there is rarely if ever such a thing as an unfair test",
  "Becoming a success is a matter of hard work, luck has little or nothing to do with it",
  "The average citizen can have an influence in government decisions",
  "When I make plans, I am almost certain that I can make them work",
  "In my case getting what I want has little or nothing to do with luck",
  "Getting people to do the right thing depends upon ability. Luck has little or nothing to do with it",
  "By taking an active part in political and social affairs the people can control world events",
  'There really is no such thing as "luck."',
  "How many friends you have depends upon how nice a person you are",
  "Most misfortunes are the result of lack of ability, ignorance, laziness, or all three",
  "With enough effort we can wipe out political corruption",
  "There is a direct connection between how hard 1 study and the grades I get",
  "It is impossible for me to believe that chance or luck plays an important role in my life",
  "People are lonely because they don't try to be friendly",
  "What happens to me is my own doing",
  "In the long run the people are responsible for bad government on a national as well as on a local level"
)

    # Negative (external locus of control)

external_items = c(
  "Many of the unhappy things in people's lives are partly due to bad luck",
  "There will always be wars, no matter how hard people try to prevent them",
  "Unfortunately, an individual's worth often passes unrecognized no matter how hard he tries",
  "Most students don't realize the extent to which their grades are influenced by accidental happenings",
  "Without the right breaks one cannot be an effective leader",
  "No matter how hard you try some people just don't like you",
  "I have often found that what is going to happen will happen",
  "Many times exam questions tend to be so unrelated to course work that studying in really useless",
  "Getting a good job depends mainly on being in the right place at the right time",
  "This world is run by the few people in power, and there is not much the little guy can do about it",
  "It is not always wise to plan too far ahead because many things turn out to- be a matter of good or bad fortune anyhow",
  "Many times we might just as well decide what to do by flipping a coin",
  "Who gets to be the boss often depends on who was lucky enough to be in the right place first",
  "As far as world affairs are concerned, most of us are the victims of forces we can neither understand, nor control",
  "Most people don't realize the extent to which their lives are controlled by accidental happenings",
  "It is hard to know whether or not a person really likes you",
  "In the long run the bad things that happen to us are balanced by the good ones",
  "It is difficult for people to have much control over the things politicians do in office",
  "Sometimes I can't understand how teachers arrive at the grades they give",
  "Many times I feel that I have little influence over the things that happen to me",
  "There's not much use in trying too hard to please people, if they like you, they like you",
  "Sometimes I feel that I don't have enough control over the direction my life is taking",
  "Most of the time I can't understand why politicians behave the way they do"
)

internal_bge <- textEmbed(
  internal_items,
  model = "BAAI/bge-base-en-v1.5", # model name
  tokens_select = "[CLS]", # select only [CLS] token embedding
  layers = -1,  # last layer
  dim_name = FALSE,
  keep_token_embeddings = FALSE
)
internal_bge <- internal_bge$texts[[1]]

external_bge <- textEmbed(
  external_items,
  model = "BAAI/bge-base-en-v1.5", # model name
  tokens_select = "[CLS]", # select only [CLS] token embedding
  layers = -1,  # last layer
  dim_name = FALSE,
  keep_token_embeddings = FALSE
)
external_bge <- external_bge$texts[[1]]

# 3. Compute Anchored Vector

  # Aggregate questionnaire embeddings
  internal_bge <- apply(as.matrix(internal_bge), 2, mean)
  external_bge <- apply(as.matrix(external_bge), 2, mean)
  
  saveRDS(internal_bge, "example_data/internal_bge.rds")
  saveRDS(external_bge, "example_data/external_bge.rds")
  
  # Anchored vector
  loc_anchored <- internal_bge - external_bge
  
# 4. Calculate Distance Metrics
  
  reddit_emotion_loc <- reddit_emotion_bge |> 
    rowwise() |> 
    mutate(loc = dot_prod(c_across(Dim1:Dim768), loc_anchored)) |> 
    pull(loc)
  
# 5. Test Hypothesis

t.test(
  x = reddit_emotion_loc[reddit_emotion$subreddit=="depression"],
  y = reddit_emotion_loc[reddit_emotion$subreddit=="TodayIamHappy"]
)
# r/TodayIamHappy has more internal (as opposed to external) locus of control
