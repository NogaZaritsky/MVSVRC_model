import math
import torch


def numerical_derivative(f, t, h=1e-5):
    return (f(t + h) - f(t - h)) / (2 * h)


def custom_integral(coefficients, t_values, q=1):
    integral_sum = torch.tensor(0.0, dtype=torch.float32)

    for i in range(len(coefficients)):
        coeffs = coefficients[i]

        values = []
        for t in t_values:
            f_t = q * uncertainty_function_at_t(coeffs, t)
            f_prime_t = numerical_derivative(lambda t_val: q * uncertainty_function_at_t(coeffs, t_val), t)

            cos_t_squared = torch.cos(t) ** 2
            sin_2t = torch.sin(2 * t)

            integrand = 0.5 * f_t * f_prime_t * sin_2t + cos_t_squared * f_t ** 2
            values.append(integrand)

        values = torch.stack(values)
        integral = torch.trapz(values, t_values)
        integral_sum = integral_sum + integral

    return integral_sum

def uncertainty_function_at_t(coefficients, t):
    value_at_t = coefficients[0]
    m = int((len(coefficients) - 1) / 2)
    for i in range(1, m + 1):
        value_at_t = value_at_t + coefficients[i] * torch.sin(i * t) + coefficients[i + m] * torch.cos(i * t)
    return torch.exp(value_at_t)


def uncertainty_loss(coefficients, y_true, mean_pred, lamda=0.1):
    delta = y_true - mean_pred
    delta_x = delta[:, 0]
    delta_y = delta[:, 1]
    angles = torch.atan2(delta_y, delta_x)

    uncertainty_values = torch.stack([
        uncertainty_function_at_t(coefficients[i], angles[i]) for i in range(len(angles))
    ])
    distance_to_true = torch.norm(y_true - mean_pred, dim=1)
    first = torch.where(uncertainty_values < distance_to_true,
                        distance_to_true - uncertainty_values,
                        torch.zeros_like(distance_to_true))
    first_sum = first.sum()

    num_steps = 50
    t_values = torch.linspace(0, 2 * math.pi, num_steps)
    integral_sum = custom_integral(coefficients, t_values)

    coefficients_first_sum = torch.sum(coefficients, dim=0)[0]

    return first_sum + lamda * integral_sum + 0.1 * lamda * coefficients_first_sum
