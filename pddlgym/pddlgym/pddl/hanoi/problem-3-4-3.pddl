(define (problem hanoi1)
  (:domain hanoi)
  (:objects peg1 peg2 peg3 peg4 d2 d3 d4)
  (:init
   (smaller peg1 d2) (smaller peg1 d3) (smaller peg1 d4)
   (smaller peg2 d2) (smaller peg2 d3) (smaller peg2 d4)
   (smaller peg3 d2) (smaller peg3 d3) (smaller peg3 d4)
   (smaller peg4 d2) (smaller peg4 d3) (smaller peg4 d4)

   (smaller d3 d2)
   (smaller d4 d2) (smaller d4 d3)

   (clear peg1) (clear peg2) (clear peg3)
   (on d4 peg4) (on d3 d4) (on d2 d3)
  )
  (:goal (and (on d4 peg3) (on d3 d4) (on d2 d3)))
  )