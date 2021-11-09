library(ROSE)
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

# Create SVM specification 
svm_specification <- svm_rbf() %>%
  set_mode('classification') %>%
  set_engine('kernlab')

# Create cross validation folds
set.seed(42)
sms_folds <- vfold_cv(sms_balanced_training)

# Create new workflow for CV
svm_workflow <- workflow() %>%
  add_recipe(sms_recipe) %>%
  add_model(svm_specification)

# Fit resampled CV folds then save or load existing file
fit_path = 'fit.rds'

if (file.exists(fit_path)) {
  svm_resampled <- readRDS(fit_path)
} else {
  options(tidymodels.dark=TRUE) # So I can actually what is going on
  svm_resampled <- fit_resamples(
    svm_workflow,
    sms_folds,
    control=control_resamples(save_pred=TRUE, verbose=TRUE)
  )

  saveRDS(svm_resampled, 'fit.rds')
}

# Evaluate performance on training set
svm_resampled_metrics <- collect_metrics(svm_resampled)
svm_resampled_predictions <- collect_predictions(svm_resampled)

# Don't forget to try it with unbalanced original
print(svm_resampled_metrics)
print(svm_resampled_predictions)
