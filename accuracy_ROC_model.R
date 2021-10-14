# function for the model resulting 
# from the train() function
# calculates: accuracy, spec, sens, AUC, balanded acc

accuracy_ROC_model <- function(model, 
                               data, 
                               target_variable = "UCURNINS",
                               predicted_class = "Yes") {
  
  require(pROC)
  # generates probabilities for "predicted_class"
  
  forecasts_p <- predict(model, data,
                         type = "prob")[, predicted_class]
  
  # and predicted category itselt
  if (any(class(model) == "train")) forecasts_c <- predict(model, data) else
     forecasts_c <- predict(model, data, type = "class")
  
  # real values - pull() converts tibble into a vector
  
  real <- (data[, target_variable]) %>% pull
  
  # area under ROC curve
  AUC <- roc(predictor = forecasts_p,
             response = real)
  
  # confussion matrix and its measures
  
  table <- confusionMatrix(forecasts_c,
                           real,
                           predicted_class) 
  # lets collect all for final result
  result <- c(table$overall[1], # Accuracy
              table$byClass[c(1:2, 11)], # sens, spec, balanced
              ROC = AUC$auc)
  
  return(result)
  
}
