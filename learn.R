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

# Create workflow
sms_workflow <- workflow() %>% add_recipe(sms_recipe)

# Create SVM specification 
svm_specification <- svm_rbf() %>%
  set_mode('classification') %>%
  set_engine('kernlab')

# Create cross validation folds
set.seed(42)
sms_folds <- vfold_cv(sms_training)

# Create new workflow for CV
svm_workflow <- workflow() %>%
  add_recipe(sms_recipe) %>%
  add_model(svm_specification)

# Evaluate performance on training set
print('Evaluating performance on training set:')

# Fit resampled CV folds then save or load existing file
fit_cv_path <- 'fit-cv.rds'

if (file.exists(fit_cv_path)) {
  svm_resampled <- readRDS(fit_cv_path)
} else {
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
  final_grid <- grid_regular(
    penalty(range=c(-4, 0)),
    max_tokens(range=c(1e3, 3e3)),
    levels = c(penalty=20, max_tokens = 3)
  )
  
  set.seed(42)

  tune_resampled <- tune_grid(
    svm_workflow,
    sms_folds,
    grid=final_grid,
    metrics=metrics,
    control=control_grid(verbose=TRUE)
  )
  
  choose_accuracy <- tune_resampled %>% select_best(metric='accuracy')
  final_workflow <- finalize_workflow(svm_workflow, choose_accuracy)
  final_fit <- last_fit(final_workflow, sms_split)

  saveRDS(final_fit, fit_final_path)
}

print(collect_metrics(final_fit))
print(collect_predictions(final_fit))
