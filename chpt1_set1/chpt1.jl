using Gadfly, DataFrames, Distributions

using pla


function simple_run(N, problem_str)
    R = 100.0
    d = 2
    w_actual, f, D = generate_data_set(N, d, R)
    iterations = 0
    w_estimate = simple_perceptron(D, (x, y, w)->iterations+=1)
    println("iterations = $iterations")
    p = plot(
        layer_data(D, 1),
        layer_data(D, -1),
        layer_w_actual(w_actual, R),
        layer_w_estimate(w_estimate, R),
        Scale.x_continuous(minvalue=-R, maxvalue=R),
        Scale.y_continuous(minvalue=-R, maxvalue=R)
    )
    draw(PDF("problem__$problem_str.pdf", 6inch, 6inch), p)
end


function iterations_over_rand_order(N, d, num_runs=100)
    R = 100.0
    w, f, D = generate_data_set(N, d, R)

    iteration_spread = Int64[]
    for i = 1:num_runs
        iterations = 0
        w = rand_order_perceptron(D, (x, y, w)->iterations+=1)
        push!(iteration_spread, iterations)
    end
    println(iteration_spread)

    iteration_spread
end


function large_dim_run(d, N)
    R = 100.0
    w, f, D = generate_data_set(N, d, R)
    iterations = 0
    w = simple_perceptron(D, (x, y, w)->iterations+=1)
    iterations
end


# Problem 1.4 a - e
#=simple_run(20, "1_4b")=#
#=simple_run(20, "1_4c")=#
#=simple_run(100, "1_4d")=#
#=simple_run(1000, "1_4e")=#

# Problem 1.4 f, g
#=iteration_spread = iterations_over_rand_order(100, 10, 1000)=#
#=p = plot(x=iteration_spread, Geom.histogram, Guide.xlabel("Iterations"), Guide.ylabel("Count"))=#
#=draw(PDF("problem_1_4g.pdf", 6inch, 6inch), p)=#

# Problem 1.4h
dimensions = 2:10
runs_per_dim = 100
num_data_points = 100

iterations_per_dim = Int64[large_dim_run(d, num_data_points) for i = 1:runs_per_dim, d = dimensions]
y = vec(mean(iterations_per_dim, 1))
ystd = vec(std(iterations_per_dim, 1))
ymins = y .- (1.96*ystd/sqrt(runs_per_dim))
ymaxs = y .+ (1.96*ystd/sqrt(runs_per_dim))

p = plot(
    x=dimensions,
    y=y,
    ymin=ymins,
    ymax=ymaxs,
    Geom.line,
    Geom.errorbar,
    Guide.xlabel("Dimensions"),
    Guide.ylabel("Mean Iterations")
)

draw(PDF("problem_1_4h.pdf", 6inch, 6inch), p)


# Problemm 1.6

for mu = (0.05, 0.5, 0.8)
    d = Binomial(10, mu)
    mu0 = pdf(d, 0)

    println("For mu = $mu")
    println(" a) $mu0")
    println(" b) $(1 - (1 - mu0)^1000)")
    println(" c) $(1 - (1 - mu0)^1000000)")
    println("")
end

