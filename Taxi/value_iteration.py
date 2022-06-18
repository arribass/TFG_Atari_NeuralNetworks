world = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
acciones = [[-1, 0 ], # arriba
         [ 0, -1], # izquierda
         [ 1, 0 ], # abajo
         [ 0, 1 ]] # derecha
simbolo_accion = ['^', '<', 'v', '>']
# Recompensas:
R = -0.04
Rs = [[R, R, R, 1], [R, -1, R, -1], [R, R, R, R]]

estrategia = [[' ']*len(world[0]) for i in world]
estrategia[0][3] = '*'

p_acierto = 0.8
p_fallo = 0.1
gamma = 1/2

def valueIteration():
    value =[[0 for col in range(len(Rs[0]))] for row in range(len(Rs))]

    for row in range(len(Rs)):
        for col in range(len(Rs[0])):
            if Rs[row][col] == 1 or Rs[row][col] == -1:
                value[row][col] = Rs[row][col]


    for row in range(len(value)):
        print(value[row])


    cambios = True
    n = 0
    while cambios and n < 100:
        # Completa el algoritmo
        continue
    return value, estrategia

m, p = valueIteration()
for row in range(len(p)):
    print(p[row])