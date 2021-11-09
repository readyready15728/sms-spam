library(ROSE)
library(discrim)
library(textrecipes)
library(tidymodels)
library(tidyverse)

# Load up SMS text messages and make class a factor
sms <- read_csv('sms.csv')
sms$class <- as.factor(sms$class)

# Split into training and testing datasets 80/20
set.seed(42)
sms_split <- initial_split(sms, prop=0.8, strata=class)
sms_training <- training(sms_split)
sms_test <- testing(sms_split)

# Resolve substantial class imbalance, making ham and spam messages nearly equally represented
sms_balanced_training <- as_tibble(ovun.sample(class ~ ., data=sms_training, method='both', N=nrow(sms_training))$data)

# Create preprocessing recipe for text classification
sms_recipe <- recipe(class ~ text, data=sms_balanced_training) %>% 
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens=1e3) %>%
  step_tfidf(text)

# Create workflow
sms_workflow <- workflow() %>% add_recipe(sms_recipe)

# Create naïve Bayes specification 
nb_specification <- naive_Bayes() %>%
  set_mode('classification') %>%
  set_engine('naivebayes')

# Fit training data
nb_fit <- sms_workflow %>%
  add_model(nb_specification) %>%
  fit(data=sms_balanced_training)

# Create cross validation folds
set.seed(42)
sms_folds <- vfold_cv(sms_balanced_training)

# Create new workflow for CV
nb_workflow <- workflow() %>%
  add_recipe(sms_recipe) %>%
  add_model(nb_specification)

# Fit resampled CV folds
nb_resampled <- fit_resamples(
  nb_workflow,
  sms_folds,
  control=control_resamples(save_pred=TRUE)
)

print(nb_resampled)
