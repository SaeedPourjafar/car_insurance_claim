summary_binary <- function(predicted_probs,
                           real,
                           cutoff = 0.5,
                           level_positive = "Yes",
                           level_negative = "No") {
  # save the classification table
  ctable <- confusionMatrix(as.factor(ifelse(predicted_probs > cutoff, 
                                             level_positive, 
                                             level_negative)), 
                            real, 
                            level_positive) 
  # extract selected statistics into a vector
  stats <- round(c(ctable$overall[1],
                   ctable$byClass[c(1:4, 7, 11)]),
                 5)
  # and return them as a function result
  return(stats)
}