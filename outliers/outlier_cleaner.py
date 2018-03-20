#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = sorted([
        (age, net_worth, prediction - net_worth)
        for age, net_worth, prediction in zip(ages, net_worths, predictions)
    ], key=lambda x: x[2])

    return cleaned_data[:int(len(cleaned_data) * 0.90):]
