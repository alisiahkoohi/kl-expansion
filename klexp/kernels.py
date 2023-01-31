import torch


def inner_product(x, y):
    """Compute the inner product of two vectors.
    """
    return torch.dot(x, y)


def rbf(x, y, sigma=1.0):
    """Compute the RBF kernel.
    """
    return torch.exp(-inner_product(x - y, x - y) / (2 * sigma**2))


def matern(x, y, sigma=1.0, nu=1.5):
    """Compute the Matern kernel.
    """
    r = torch.sqrt(inner_product(x - y, x - y))
    if nu == 0.5:
        return sigma**2 * torch.exp(-r)
    elif nu == 1.5:
        return sigma**2 * (1 + r) * torch.exp(-r)
    elif nu == 2.5:
        return sigma**2 * (1 + r + r**2 / 3) * torch.exp(-r)
    else:
        raise ValueError("nu must be 0.5, 1.5, or 2.5.")


def periodic(x, y, sigma=1.0, l=1.0, p=1.0):
    """Compute the periodic kernel.
    """
    return sigma**2 * torch.exp(
        -2 * torch.sin(np.pi * torch.abs(x - y) / p)**2 / l**2)


def squared_exponential(x, y, sigma=1.0, l=1.0):
    """Compute the squared exponential kernel.
    """
    return sigma**2 * torch.exp(-inner_product(x - y, x - y) / l**2)


def rational_quadratic(x, y, sigma=1.0, alpha=1.0):
    """Compute the rational quadratic kernel.
    """
    return sigma**2 * (1 + inner_product(x - y, x - y) /
                       (2 * alpha * sigma**2))**(-alpha)


def exponential(x, y, sigma=1.0, l=1.0):
    """Compute the exponential kernel.
    """
    return sigma**2 * torch.exp(-torch.sqrt(inner_product(x - y, x - y)) / l)


def cauchy(x, y, sigma=1.0, l=1.0):
    """Compute the Cauchy kernel.
    """
    return sigma**2 / (1 + inner_product(x - y, x - y) / l**2)


def linear(x, y, sigma=1.0):
    """Compute the linear kernel.
    """
    return sigma**2 * inner_product(x, y)


def polynomial(x, y, sigma=1.0, d=2):
    """Compute the polynomial kernel.
    """
    return sigma**2 * (inner_product(x, y) + 1)**d


def sigmoid(x, y, sigma=1.0, l=1.0):
    """Compute the sigmoid kernel.
    """
    return torch.tanh(sigma * inner_product(x, y) + l)