library(xgboost)
library(Matrix)
library(caret)
library(data.table)

df <- fread(input = "https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv", data.table = F)
str(df)
prop.table(table(df$V42))

df$V42 <- ifelse(df$V42 == 'RB', 0, 1)

model_matrix <- model.matrix(V42 ~ . -1, data = df)
output_vector <- df$V42
dtrain <- xgb.DMatrix(data = model_matrix, label = output_vector)

params <- list(
  booster = "gbtree"
  ,objective = "binary:logistic"
  ,eta = 0.1
  ,gamma = 0
  ,max_depth = 6
  ,min_child_weight = 1
  ,subsample = 1
  ,colsample_bytree = 1
)


xgb_cv <- xgb.cv(
  params = params
  ,data = dtrain
  ,nrounds = 500
  ,nfold = 10
  ,showsd = T
  ,stratified = T
  ,print_every_n = 5
  ,early_stopping_rounds = 10
  ,maximize = F
  ,prediction = T
  ,metrics = "logloss"
)

cv_pred <- xgb_cv$pred
cv_pred_class <- ifelse(cv_pred > 0.5, 1, 0)

confusionMatrix(
  data = cv_pred_class
  ,reference = getinfo(dtrain, 'label')
  ,positive = "1"
  ,mode = "prec_recall"
)




