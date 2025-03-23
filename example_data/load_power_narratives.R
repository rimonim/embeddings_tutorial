# Training Dataset: Kasprzyk & Calin-Jageman (2014) 

library(tidyverse)

power_mturk <- haven::read_sav("https://osf.io/download/9bk3f/") |> 
  select(Control, Power) |> 
  pivot_longer(Control:Power, names_to = "condition", values_to = "text") |> 
  filter(text != "") |> 
  mutate(doc_id = 1:n(),
         platform = "mturk")

power_prolific <- haven::read_sav("https://osf.io/download/e9sp6/") |> 
  rename(Power = Power.0) |> 
  select(Control, Power) |> 
  pivot_longer(Control:Power, names_to = "condition", values_to = "text") |> 
  filter(text != "") |> 
  mutate(doc_id = 1:n(),
         platform = "prolific")

write_csv(bind_rows(power_mturk,power_prolific), "example_data/power_narratives.csv")
