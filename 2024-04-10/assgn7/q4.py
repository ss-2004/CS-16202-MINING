def justify_bayes_probability(bayes_probs):

    max_prob_label = max(bayes_probs, key=bayes_probs.get)
    justification = f"The instance is more likely to belong to class'{max_prob_label}' with a probability of{bayes_probs[max_prob_label]:.4f}"
    return justification

justification = justify_bayes_probability(bayes_probs)
print(justification)