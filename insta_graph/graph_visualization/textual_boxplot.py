import numpy as np

def textual_boxplot(values, display_min, display_max, total_length=50):
    """Generate a textual representation of a boxplot from a list of values."""

    # Compute key statistics
    min_val = min(values)
    q1 = np.percentile(values, 25)
    median = np.percentile(values, 50)
    q3 = np.percentile(values, 75)
    max_val = max(values)

    # Adjust values to be within display range
    min_val = max(min_val, display_min)
    max_val = min(max_val, display_max)

    # Calculate relative positions
    min_pos = round(total_length * (min_val - display_min) / (display_max - display_min))
    q1_pos = round(total_length * (q1 - display_min) / (display_max - display_min))
    median_pos = round(total_length * (median - display_min) / (display_max - display_min))
    q3_pos = round(total_length * (q3 - display_min) / (display_max - display_min))
    max_pos = round(total_length * (max_val - display_min) / (display_max - display_min))

    # Build the textual representation
    plot = [' '] * total_length

    # Filling in the '-' characters
    for i in range(min_pos + 1, q1_pos):
        plot[i] = '-'
    for i in range(q3_pos + 1, max_pos):
        plot[i] = '-'

    # Setting the specific characters
    plot[min_pos] = '|'
    if q1_pos == median_pos == q3_pos:
        plot[q1_pos] = '*'
    else:
        plot[q1_pos] = '['
        plot[q3_pos] = ']'
    plot[median_pos] = '|'
    plot[max_pos] = '|'

    return str(display_min) + " " + "".join(plot) + " " + str(display_max)

# Sample usage
values = [2014, 2015, 2016, 2016, 2017, 2018, 2019, 2020, 2022]
# print(textual_boxplot(values, 2014, 2023))
