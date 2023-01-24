import numpy
from scipy.integrate import solve_ivp
from scipy import optimize

#**************************************************************************************************
# Functions
#**************************************************************************************************

# function for the ODE system we wish to solve
# z refers to the compartments
# all parameters after z refer to the parameters in the model
def model(t, z, a, b, c, d):
    u, v = z
    return [a*u-b*u*v, c*u*v - d*v]

# computes the difference between two vectors
# and takes the magnitude of the result
def vectorDifference(y1, y2):
    sum = 0
    for i in range(len(y1)):
        sum += (y1[i] - y2[i])**2
    
    sum = sum**(1/2)
    return sum

# this is our cost function
# it tries to minimize the difference between the
# noisy data and the estimated data
def errorInData(x0):
    est_sol = solve_ivp(model, time_interval, initial_c, args=x0, t_eval=m_t_eval, dense_output=True)

    est_data = []
    for i in range(len(est_sol.y[0])):
        point = []
        for j in range(len(est_sol.y)):
            if is_observable[j]:
                point.append(est_sol.y[j][i])
        est_data.append(point)

    sum = 0
    for i in range(len(est_data)):
        sum += vectorDifference(est_data[i], error_data[i])**2
    return sum

# computes the ARE scores for each parameter
def ARE_score(parameter_values):
    score = [0 for i in range(len(m_args))]
    for i in range(len(parameter_values)):
        for j in range(len(m_args)):
            score[j] += numpy.abs(parameter_values[i][j] - m_args[j])/(m_args[j])

    for i in range(len(m_args)):
        score[j] *= 100/len(parameter_values)
    return score

#**************************************************************************************************
# Main function
#**************************************************************************************************

time_interval = [0, 1]
num_of_points = 100
iterations = 100
h = (time_interval[1] - time_interval[0]) / num_of_points

m_t_eval = []
for i in range(num_of_points):
    m_t_eval.append(time_interval[0]+h*i)

m_args = (2, 10, 2, 2)
initial_c = [20, 1]
noise_variance = [0, 0.01, 0.05, 0.1, 0.2, 0.3]

is_observable = [True, False]

# solves using the true parameters
sol = solve_ivp(model, time_interval, initial_c, args=m_args, t_eval=m_t_eval, dense_output=True)

# sol.y contains all the compartments
# but we want only the observables
data = []
for i in range(len(sol.y[0])):
    point = []
    for j in range(len(sol.y)):
        if is_observable[j]:
            point.append(sol.y[j][i])
    data.append(point)

x_values = []
for variance in noise_variance:
    x_iteration = []
    for i in range(iterations):
        error_data = list.copy(data)

        # applies noise to the data
        noise = numpy.random.normal(0, variance, num_of_points)
        for k in range(len(error_data)):
            for j in range(len(error_data[0])):
                error_data[k][j] = error_data[k][j] + error_data[k][j] * noise[k]

        # fminsearch function
        print(i, ": ")
        xopt = optimize.minimize(errorInData, method='SLSQP', x0=[0.1 for i in range(len(m_args))], bounds=((0, 100) for i in range(len(m_args))))
        print(xopt.x)
        x_iteration.append(xopt.x)
    x_values.append(x_iteration)

# prints out the ARE scores for each parameter
for i in range(len(noise_variance)):
    print(noise_variance[i], ":", ARE_score(x_values[i]))