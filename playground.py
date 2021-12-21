from DAGs_Generator import DAGs_generate
from DAGs_Generator import plot_DAG

# edges,in_degree,out_degree,position = DAGs_generate('default')
edges,in_degree,out_degree,position = DAGs_generate('default',n = 30,max_out = 3,alpha = 1.5,beta = 1.0)
plot_DAG(edges,position)
print(edges,in_degree,out_degree)