
# Open a file in write mode. If the file doesn't exist, it will be created.
with open('output_file.txt', 'r') as file:

    lines = file.readlines()
    counter=28
    # Iterate through each element in the array
    for line in lines:
        # Write the element to the file, followed by a newline character
        

        with open("problem_hanoi-4-4_"+str(counter)+".pddl", 'w') as file2:

            
            
            pre = "(define (problem hanoi1) (:domain hanoi) \n" + \
            "(:objects peg1 peg2 peg3 peg4 d1 d2 d3 d4) \n" + \
            "(:init \n" + \
            "(smaller peg1 d1) (smaller peg1 d2) (smaller peg1 d3) (smaller peg1 d4) \n" + \
            "(smaller peg2 d1) (smaller peg2 d2) (smaller peg2 d3) (smaller peg2 d4) \n" + \
            "(smaller peg3 d1) (smaller peg3 d2) (smaller peg3 d3) (smaller peg3 d4) \n" + \
            "(smaller peg4 d1) (smaller peg4 d2) (smaller peg4 d3) (smaller peg4 d4) \n" + \
            "(smaller d2 d1) (smaller d3 d1) (smaller d4 d1) \n" + \
            "(smaller d3 d2) (smaller d4 d2) \n"+ \
            "(smaller d4 d3) \n"
            file2.write(pre + '\n')

            file2.write(line + '\n')

            
            suc = "  ) \n" + \
            "(:goal (and (on d2 d3) (on d3 d2))) \n" + \
            ") \n"

            file2.write(suc + '\n')

        counter+=1