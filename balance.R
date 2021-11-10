library(ROSE)
library(tidyverse)

# Write out SMS spam/ham dataset with class imbalance nearly eliminated
sms <- read_csv('sms.csv')
sms_balanced <- as_tibble(ovun.sample(class ~ ., data=sms, method='both', N=nrow(sms))$data)
write_csv(sms_balanced, 'sms-balanced.csv')
