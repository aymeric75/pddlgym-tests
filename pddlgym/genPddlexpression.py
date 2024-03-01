from itertools import permutations

# Original pegs in the expression
original_pegs = ['peg1', 'peg2', 'peg3', 'peg4']

# Generate all permutations of the original pegs
permutations_list = list(permutations(original_pegs))

# Prepare the base part of the expression without peg numbers
base_expression = [
    "(clear {})", "(clear {})", "(clear d1)", "(clear d2)",
    "(on d1 d4)", "(on d2 d3)", "(on d3 {})", "(on d4 {})"
]

# Apply each permutation to the base expression
interchanged_expressions = []
for perm in permutations_list:
    # Map the permutation to the corresponding base expression parts
    expression_with_perm = [part.format(*perm) if '{}' in part else part for part in base_expression]
    interchanged_expressions.append(' '.join(expression_with_perm))

# Show the first few interchanged expressions as an example
interchanged_expressions[:5]  # Displaying only the first 5 for brevity




# Correcting the approach to apply permutations accurately to the expression parts that include pegs

# Corrected base parts of the expression that need peg number replacements
base_parts_needing_replacement = ["(clear {})", "(clear {})", "(on d2 {})", "(on d4 {})"]

# Apply each permutation to the parts of the expression that need replacement
corrected_interchanged_expressions = []
for perm in permutations_list:
    # Apply the permutation only to the parts of the expression with pegs
    replaced_parts = [part.format(peg) for part, peg in zip(base_parts_needing_replacement, perm)]
    # Reconstruct the full expression with the permuted parts
    full_expression = ' '.join(["(clear d1)", "(clear d2)", "(on d1 d2)", "(on d3 d4)"] + replaced_parts)
    corrected_interchanged_expressions.append(full_expression)

# Show the first few corrected interchanged expressions as an example
corrected_interchanged_expressions[:5]  # Displaying only the first 5 for brevity


print(corrected_interchanged_expressions)


# Define your array
array = ['Element1', 'Element2', 'Element3', 'Element4']

# Open a file in write mode. If the file doesn't exist, it will be created.
with open('output_file.txt', 'w') as file:
    # Iterate through each element in the array
    for element in corrected_interchanged_expressions:
        # Write the element to the file, followed by a newline character
        file.write(element + '\n')
