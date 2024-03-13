(define (problem blocks)
    (:domain blocks)
    (:objects 
        b - block
        a - block
        c - block
        robot - robot
    )
    (:init 
        (clear c) 
        (clear a) 
        (clear b) 
        (ontable c) 
        (ontable a)
        (ontable b) 
        (handempty robot)

        ; action literals
        (pickup a)
        (putdown a)
        (unstack a)
        (stack a b)
        (stack a c)
        (pickup b)
        (putdown b)
        (unstack b)
        (stack b a)
        (stack b c)
        (pickup c)
        (putdown c)
        (unstack c)
        (stack c b)
        (stack c a)


    )
    (:goal (and (on a b) (on b c) (ontable c)))
)
