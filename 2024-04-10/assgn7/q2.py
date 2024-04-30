# Compute the probability of the class labels (buys_computer=”yes” or buys_computer= ”no”) for each
# nominal value in an attribute.

def compute_class_label_probabilities(df, attribute):
    
    class_label_probs = {}
    unique_values = df[attribute].unique()
    total_samples = len(df)
    for value in unique_values:
        value_df = df[df[attribute] == value]
        yes_prob = value_df[value_df['buys_computer'] == 'yes'].shape[0] / total_samples
        no_prob = value_df[value_df['buys_computer'] == 'no'].shape[0] / total_samples
        class_label_probs[value] = {'yes': yes_prob, 'no': no_prob}
    return class_label_probs

class_label_probs = compute_class_label_probabilities(df, 'age')
print(class_label_probs)
