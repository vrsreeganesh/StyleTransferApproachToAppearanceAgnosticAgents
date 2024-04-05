# train00.py
Aim:
    Bring in style transfer to transfuser
Note:
    for now we just style transferred the image before feeding to training module. 
Result: 
    sucesful. We're freezing and forking cause I don't wanna mess this up.

# train01.py
Aim:
    Bring in style transfer and let the model train with it.
Note:
    So the idea right now is put the model to train and see how well it does. 
    Its gonna be a batch job, obviously. 
    We'll put in a switch like thing so that a percentage number of inputs are style transferred.
    This maneuver will save us shit tons of time. 
    