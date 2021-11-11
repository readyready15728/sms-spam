library(textrecipes)
library(tidymodels)
library(tidyverse)

# So I can actually what is going on during logging later on
options(tidymodels.dark=TRUE)

# Defining metrics for both training and testing
metrics <- metric_set(accuracy, sensitivity, specificity, roc_auc)

# Load up balanced SMS text messages and make class a factor
sms <- read_csv('sms-balanced.csv')
sms$class <- as.factor(sms$class)

# Split into training and testing datasets 80/20
set.seed(42)
sms_split <- initial_split(sms, prop=0.8, strata=class)
sms_training <- training(sms_split)
sms_test <- testing(sms_split)

# Create preprocessing recipe for text classification
sms_recipe <- recipe(class ~ text, data=sms_training) %>% 
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens=1e3) %>%
  step_tfidf(text)

# Create SVM specification 
svm_specification <- svm_rbf() %>%
  set_mode('classification') %>%
  set_engine('kernlab')

# Create new workflow for CV
svm_workflow <- workflow() %>%
  add_recipe(sms_recipe) %>%
  add_model(svm_specification)

# Create cross validation folds
set.seed(42)
sms_folds <- vfold_cv(sms_training)

# Evaluate performance on training set
print('Evaluating performance on training set:')

# Fit resampled CV folds then save or load existing file
fit_cv_path <- 'fit-cv.rds'

if (file.exists(fit_cv_path)) {
  svm_resampled <- readRDS(fit_cv_path)
} else {
  svm_workflow <- prepare_workflow(1e3)
  svm_resampled <- fit_resamples(
    svm_workflow,
    sms_folds,
    control=control_resamples(save_pred=TRUE, verbose=TRUE),
    metrics=metrics
  )

  saveRDS(svm_resampled, fit_cv_path)
}

print(collect_metrics(svm_resampled))
print(collect_predictions(svm_resampled))

# Evaluate performance on test set, either by loading and existing file or
# generating the final fit and saving
print('Evaluating performance on test set:')

fit_final_path <- 'fit-final.rds'

if (file.exists(fit_final_path)) {
  final_fit <- readRDS(fit_final_path)
} else {
  final_fit <- last_fit(svm_workflow, sms_split, metrics=metrics)

  saveRDS(final_fit, fit_final_path)
}

print(collect_metrics(final_fit))
print(collect_predictions(final_fit))
