module pla

using Gadfly, DataFrames

export simple_perceptron, rand_order_perceptron, generate_data_set, layer_w_actual, layer_w_estimate, layer_data


function properly_categorized(x, y_actual, w)
    y_estimated = isposdef(sum(x.*w)) ? 1 : -1
    y_estimated == y_actual
end


function base_perceptron(D, update, next, done)
    w = zeros(D[1][1])
    i = 1
    while !done(i, w, D)
        x, y = D[i]
        if properly_categorized(x, y, w)
            i = next(false, i, w, D)
        else
            w = update(x, y, w)
            i = next(true, i, w, D)
        end
    end
    w
end

function simple_perceptron(D, each_iter=(x, y, w) -> nothing)
    N = length(D)

    update(x, y, w) = (each_iter(x, y, w); w + x*y)
    next(was_update, i, w, D) = was_update ? 1 : i + 1
    done(i, w, D) = i == N

    base_perceptron(D, update, next, done)
end


function rand_order_perceptron(D, each_iter=(x, y, w) -> nothing)
    N = length(D)

    update(x, y, w) = (each_iter(x, y, w); w + x*y)
    next(was_update, i, w, D) = rand(1:N)
    done(i, w, D) = all(d -> properly_categorized(d[1], d[2], w), D)

    base_perceptron(D, update, next, done)
end


function adaline_perceptron(D, each_iter=(x, y, w) -> nothing)
    N = length(D)

    update(x, y, w) = (each_iter(x, y, w); w + x*y)
    next(was_update, i, w, D) = rand(1:N)
    done(i, w, D) = all(d -> properly_categorized(d[1], d[2], w), D)

    base_perceptron(D, update, next, done)
end


function rand_on_n_surface(n)
    x::Array{Float64,1} = []
    r = 0
    while r == 0
        x = randn(n)
        r = sqrt(dot(x, x))
    end
    x/r
end


function rand_in_unit_n_ball(n)
    x_surface = rand_on_n_surface(n)
    rand()^(1/n)*x_surface
end


function rand_w(n, R)
    x = R*rand_in_unit_n_ball(n)
    offset = norm(x)
    x_norm = x/offset
    [offset; x_norm]
end


function generate_data_set(N, d, R)
    w = rand_w(d, R/2) # divide by 2 to avoid boring target functions
    away_side_is_positive = randbool()
    if away_side_is_positive
        f(x) = dot(w, x) > 0 ? 1.0 : -1.0
    else
        f(x) = dot(w, x) < 0 ? 1.0 : -1.0
    end

    x_vec = [[1, R*rand_in_unit_n_ball(d)] for i = 1:N]
    D = [(x, f(x)) for x in x_vec]
    w, f, D
end


function w_to_line(w, R)
    aa = -w[2]/w[3]
    bb = -w[1]/w[3]
    x_vec = [(-R - bb)/aa, (R - bb)/aa]
    y_vec = [-R, R]
    x_vec, y_vec
end


function layer_w_estimate(w, R)
    x_vec, y_vec = w_to_line(w, R)
    layer(x=x_vec, y=y_vec, Theme(default_color=color("gray")), Geom.line)
end


function layer_w_actual(w, R)
    x_vec, y_vec = w_to_line(w, R)
    layer(x=x_vec, y=y_vec, Theme(default_color=color("black")), Geom.line)
end


function layer_data(D, yval)
    x = map(d -> d[1][2], filter(d -> d[2] == yval, D))
    y = map(d -> d[1][3], filter(d -> d[2] == yval, D))

    c = color(yval == 1 ? "red" : "blue") # HACK
    layer(x=x, y=y, Theme(default_color=c), Geom.point)
end

end
