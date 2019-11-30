import mlrose
import numpy as np

def TSP(ln,coords):

    coord=list(coords).copy()
    iters=0
    distance_sum=0
    dic={}
    points={}
    while len(coord)>0:
        iters+=1
        distance=0
        n=[]
        if len(coord)<ln:
            n=[idx for idx,i in enumerate(coord)]
        else:
            l=ln
            while len(n) != l:
                j=np.random.randint(1,len(coord))-1 
                if j not in n:
                    n.append(j)
#         print(n)
#         try:
        coords_list=[i for idx,i in enumerate(coord) if idx in n]
#             print(coords_list)

        for i in coords_list:
            coord.remove(i)
        # Initialize fitness function object using coords_list
        fitness_coords = mlrose.TravellingSales(coords = coords_list)

        # Define optimization problem object
        problem_fit = mlrose.TSPOpt(length = len(coords_list), fitness_fn = fitness_coords, maximize=False)

        best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = None)

        for i,j in zip(best_state[:-1],best_state[1:]):
            distance+=np.sqrt((coords_list[i][0]-coords_list[j][0])**2+(coords_list[i][1]-coords_list[j][1])**2)
        points[iters]=[coords_list[i] for i in best_state]
#             print(coords_list)     
#             print('d')
        # print(iters)
        # print(distance)
        dic[iters]=distance
        distance_sum+=(distance)
        # print(distance_sum)
            # best_fitness
#         except:
#             pass
        # print(len(coord))
    return dic,iters,distance_sum,points
# asp=[[np.random.randint(-100,100),np.random.randint(-100,100)] for i in range(10)]
# print(asp)
# iters,distance=TSP(3,asp)
# print(iters)
# print(distance)