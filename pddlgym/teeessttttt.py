import time

for i in range(99):
    time.sleep(2)
    with open("essai.txt", "r") as f:
        first_line = f.readline()

        # Process the first line (e.g., print it)
        print("First line:", first_line)