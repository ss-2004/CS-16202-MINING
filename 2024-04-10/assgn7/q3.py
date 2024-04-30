# Using Bayes’ theorem compute the probability of instance X = (age = youth, income = medium, student
# = yes, credit_rating = fair) against class label (buys_computer=”yes” and buys_computer=”no”).

def compute_bayes_probability(df, instance_values, class_label):
    
    total_samples = len(df)
    class_counts = df[class_label].value_counts()
    class_probs = class_counts / total_samples
    bayes_probs = {}
    for label in class_probs.index:
        prob = class_probs[label]
        for attr, value in instance_values.items():
            subset = df[df[attr] == value]
            subset_count = subset[class_label].value_counts().get(label, 0)
            
            prob *= (subset_count + 1) / (class_counts[label] + len(subset[attr].unique()))  
            bayes_probs[label] = prob
    return bayes_probs

instance_values = {'age': 'youth', 'income': 'medium', 'student': 'yes', 'credit_rating': 'fair'}

bayes_probs = compute_bayes_probability(df, instance_values, 'buys_computer')
print(bayes_probs)
